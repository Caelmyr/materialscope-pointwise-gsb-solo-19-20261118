import hashlib
import io
import json
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import contracts as c
from .numerics import percentile_limits, label_palette
from .model import spectral_band


class PreviewCache:
    def __init__(self, limit=24000000):
        self.limit = limit
        self.bytes = 0
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.entries:
                return None
            result = self.entries.pop(key)
            self.entries[key] = result
            return result

    def put(self, key, data):
        with self.lock:
            if key in self.entries:
                self.bytes -= len(self.entries.pop(key))
            if len(data) > self.limit:
                return
            while self.entries and self.bytes + len(data) > self.limit:
                _, removed = self.entries.popitem(last=False)
                self.bytes -= len(removed)
            self.entries[key] = data
            self.bytes += len(data)

    def status(self):
        with self.lock:
            return {'entries': len(self.entries), 'bytes': self.bytes, 'limit': self.limit}


def colormap(values, name):
    values = np.clip(values, 0, 1)
    if name == 'gray':
        return np.repeat(values[:, :, None], 3, axis=2)
    if name == 'heat':
        red = np.clip(values * 3, 0, 1)
        green = np.clip(values * 3 - 1, 0, 1)
        blue = np.clip(values * 3 - 2, 0, 1)
        return np.stack([red, green, blue], axis=2)
    if name == 'diverging':
        lower = values <= 0.5
        alpha = np.where(lower, values * 2, (1 - values) * 2)
        red = np.where(lower, alpha, 1)
        green = alpha
        blue = np.where(lower, 1, alpha)
        return np.stack([red, green, blue], axis=2)
    anchors = np.asarray([[0.07, 0.12, 0.26], [0.05, 0.38, 0.54], [0.12, 0.66, 0.57], [0.66, 0.85, 0.38], [0.98, 0.91, 0.36]])
    positions = np.linspace(0, 1, len(anchors))
    return np.stack([np.interp(values, positions, anchors[:, channel]) for channel in range(3)], axis=2)


def validate_view(cube, options):
    options = c.object_value(options, '显示参数')
    mode = c.choice(options.get('mode', 'rgb' if cube.count >= 3 else 'band'), '显示方式', ['rgb', 'band', 'validity', 'labels'])
    low, high = c.percent_range(options.get('low', 2), options.get('high', 98))
    gamma = c.number(options.get('gamma', 1), '显示伽马', 0.1, 5)
    band = c.integer(options.get('band', 0), '显示波段', 0, cube.count - 1)
    default_rgb = [min(2, cube.count - 1), min(1, cube.count - 1), 0]
    rgb = c.array_value(options.get('rgb', default_rgb), 'RGB波段', 3, 3)
    rgb = [c.integer(item, 'RGB波段', 0, cube.count - 1) for item in rgb]
    palette = c.choice(options.get('palette', 'ocean'), '色带', ['ocean', 'gray', 'heat', 'diverging'])
    background = c.choice(options.get('background', 'checker'), '背景', ['checker', 'black', 'transparent'])
    maximum = c.integer(options.get('maximum', 900), '预览边长', 64, 1600)
    return {'mode': mode, 'low': low, 'high': high, 'gamma': gamma, 'band': band, 'rgb': rgb, 'palette': palette, 'background': background, 'maximum': maximum}


def default_view(cube):
    if cube.domain == 'labels':
        return validate_view(cube, {'mode': 'labels'})
    if cube.count >= 3:
        try:
            rgb = [spectral_band(cube, wave, 30000) for wave in (650, 550, 450)]
        except c.Invalid:
            rgb = [2, 1, 0]
        return validate_view(cube, {'mode': 'rgb', 'rgb': rgb})
    return validate_view(cube, {'mode': 'band'})


def render_array(cube, options):
    options = validate_view(cube, options)
    mode = options['mode']
    if mode == 'validity':
        fraction = cube.valid.mean(axis=2)
        rgb = colormap(fraction, 'ocean')
        valid = np.ones(cube.shape[:2], dtype=bool)
        ranges = [[0, 1]]
    elif mode == 'labels':
        labels = cube.values[:, :, options['band']].astype(int)
        if np.any(labels[cube.valid[:, :, options['band']]] < 0):
            raise c.Invalid('标签显示不接受负数')
        maximum_label = min(int(labels.max()), 1000000)
        palette = label_palette(min(maximum_label + 1, 1001))
        rgb = palette[np.mod(np.maximum(labels, 0), len(palette))].astype(float) / 255
        valid = cube.valid[:, :, options['band']]
        ranges = [[0, maximum_label]]
    else:
        selected = options['rgb'] if mode == 'rgb' else [options['band']]
        channels = []
        masks = []
        ranges = []
        for band in selected:
            values = cube.values[:, :, band]
            mask = cube.valid[:, :, band]
            low, high = percentile_limits(values, mask, options['low'], options['high'])
            normalized = np.clip((values - low) / (high - low), 0, 1) ** (1 / options['gamma'])
            channels.append(normalized)
            masks.append(mask)
            ranges.append([low, high])
        valid = np.logical_and.reduce(masks)
        if mode == 'rgb':
            rgb = np.stack(channels, axis=2)
        else:
            rgb = colormap(channels[0], options['palette'])
    rgb = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    if options['background'] == 'checker':
        yy, xx = np.indices(cube.shape[:2])
        checker = np.where(((xx // 8 + yy // 8) % 2) == 0, 32, 45).astype(np.uint8)
        rgb[~valid] = checker[~valid, None]
        alpha[:] = 255
    elif options['background'] == 'black':
        rgb[~valid] = 0
        alpha[:] = 255
    else:
        rgb[~valid] = 0
    rgba = np.dstack([rgb, alpha])
    return rgba, {'view': options, 'ranges': ranges, 'valid_pixels': int(valid.sum())}


def render_png(cube, options, cache=None):
    options = validate_view(cube, options)
    key = hashlib.sha256((cube.fingerprint() + json.dumps(options, sort_keys=True)).encode()).hexdigest()
    if cache:
        found = cache.get(key)
        if found is not None:
            return found
    rgba, report = render_array(cube, options)
    image = Image.fromarray(rgba)
    image.thumbnail((options['maximum'], options['maximum']), Image.Resampling.NEAREST if cube.domain == 'labels' else Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format='PNG')
    data = output.getvalue()
    if cache:
        cache.put(key, data)
    return data


def overlay_regions(cube, options, regions):
    rgba, report = render_array(cube, options)
    image = Image.fromarray(rgba)
    draw = ImageDraw.Draw(image)
    line_width = max(1, min(cube.width, cube.height) // 150)
    for region in regions:
        box = region['bounds']
        color = region['color']
        if region['kind'] == 'ellipse':
            draw.ellipse([box[0], box[1], box[2] - 1, box[3] - 1], outline=color, width=line_width)
        else:
            draw.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], outline=color, width=line_width)
        label = region['id'][-4:]
        draw.text((box[0] + 3, box[1] + 3), label, fill=color)
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def band_contact_sheet(cube, options):
    columns = c.integer(options.get('columns', 4), '列数', 1, 8)
    tile_size = c.integer(options.get('tile_size', 180), '单格尺寸', 80, 400)
    count = cube.count
    rows = (count + columns - 1) // columns
    sheet = Image.new('RGB', (columns * tile_size, rows * (tile_size + 30)), '#15202b')
    draw = ImageDraw.Draw(sheet)
    for index, band in enumerate(cube.bands):
        view = {'mode': 'band', 'band': index, 'maximum': tile_size, 'palette': options.get('palette', 'gray')}
        tile = Image.open(io.BytesIO(render_png(cube, view))).convert('RGB')
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + 30)
        sheet.paste(tile, (x + (tile_size - tile.width) // 2, y + (tile_size - tile.height) // 2))
        wave = band.get('wavelength_nm')
        label = f'B{index}: {wave:g} nm' if wave is not None else f'Band {index}'
        draw.text((x + 8, y + tile_size + 7), label, fill='#deebf5')
    output = io.BytesIO()
    sheet.save(output, format='PNG')
    return output.getvalue()


def compare_preview(first, second, options):
    if first.shape[:2] != second.shape[:2]:
        raise c.Invalid('图像比较需要相同像素尺寸')
    a, _ = render_array(first, options.get('first', default_view(first)))
    b, _ = render_array(second, options.get('second', default_view(second)))
    mode = c.choice(options.get('mode', 'split'), '比较模式', ['split', 'checker'])
    yy, xx = np.indices(first.shape[:2])
    if mode == 'split':
        position = c.number(options.get('position', 0.5), '分界位置', 0, 1)
        mask = xx >= first.width * position
    else:
        size = c.integer(options.get('size', 24), '棋盘格大小', 4, 200)
        mask = (xx // size + yy // size) % 2 == 1
    result = np.where(mask[:, :, None], b, a)
    output = io.BytesIO()
    Image.fromarray(result).save(output, format='PNG')
    return output.getvalue()


def _grayscale_band(cube, band):
    values = cube.values[:, :, band].astype(float)
    valid = cube.valid[:, :, band]
    if not valid.any():
        return np.zeros(cube.shape[:2], dtype=float), valid
    low, high = percentile_limits(values, valid, 2, 98)
    normalized = np.clip((values - low) / max(high - low, 1e-12), 0, 1)
    return normalized, valid


def registration_blend(cube, reference_band, moving_band, options, warp_maps=None):
    """参考带与待配带叠看；warp_maps 给定（map_x,map_y）时叠加的是按当前审阅变形后的待配带。"""
    from .numerics import bilinear_sample
    reference_band = c.integer(reference_band, '参考波段', 0, cube.count - 1)
    moving_band = c.integer(moving_band, '待配波段', 0, cube.count - 1)
    mode = c.choice(options.get('mode', 'color'), '叠看模式', ['color', 'difference', 'grayscale'])
    alpha = c.number(options.get('alpha', 0.5), '叠看透明度', 0.05, 0.95)
    reference_gray, reference_valid = _grayscale_band(cube, reference_band)
    if warp_maps is not None:
        map_x, map_y = warp_maps
        moving_values, moving_valid = bilinear_sample(
            cube.values[:, :, moving_band], cube.valid[:, :, moving_band], map_x, map_y, strict=True,
        )
        low, high = percentile_limits(moving_values, moving_valid, 2, 98)
        moving_gray = np.clip((moving_values - low) / max(high - low, 1e-12), 0, 1)
    else:
        moving_gray, moving_valid = _grayscale_band(cube, moving_band)
    common = reference_valid & moving_valid
    height, width = reference_gray.shape
    if mode == 'difference':
        diff = np.abs(reference_gray - moving_gray)
        rgb = colormap(diff, 'heat')
    elif mode == 'grayscale':
        mixed = reference_gray * (1 - alpha) + moving_gray * alpha
        rgb = np.repeat(mixed[:, :, None], 3, axis=2)
    else:
        red = moving_gray
        green_blue = reference_gray
        rgb = np.stack([red, green_blue, green_blue], axis=2)
        rgb = np.clip(rgb, 0, 1)
    rgb = np.round(rgb * 255).astype(np.uint8)
    yy, xx = np.indices((height, width))
    checker = np.where(((xx // 8 + yy // 8) % 2) == 0, 32, 45).astype(np.uint8)
    rgb[~common] = checker[~common, None]
    rgba = np.dstack([rgb, np.full((height, width), 255, dtype=np.uint8)])
    image = Image.fromarray(rgba)
    maximum = c.integer(options.get('maximum', 900), '预览边长', 64, 1600)
    image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue(), {'common_valid_pixels': int(common.sum()), 'warped_preview': warp_maps is not None}


def annotation_overlay(cube, session, options=None):
    """在参考带灰度图上绘制对应点、候选与独立材料区域，供审阅与重开核对。"""
    options = options or {}
    band = c.integer(options.get('band', session['reference_band']), '绘制波段', 0, cube.count - 1)
    gray, valid = _grayscale_band(cube, band)
    rgb = np.repeat((gray * 255)[:, :, None], 3, axis=2).astype(np.uint8)
    yy, xx = np.indices(cube.shape[:2])
    checker = np.where(((xx // 8 + yy // 8) % 2) == 0, 32, 45).astype(np.uint8)
    rgb[~valid] = checker[~valid, None]
    image = Image.fromarray(rgb).convert('RGB')
    draw = ImageDraw.Draw(image)
    line_width = max(1, min(cube.width, cube.height) // 200)
    for region in session.get('regions', []):
        color = region['color']
        shape = region['shape']
        if region['kind'] == 'rectangle':
            x0, y0, x1, y1 = shape
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=line_width + 1)
        elif region['kind'] == 'ellipse':
            x0, y0, x1, y1 = shape
            draw.ellipse([x0, y0, x1 - 1, y1 - 1], outline=color, width=line_width + 1)
        else:
            draw.line([tuple(point) for point in shape] + [tuple(shape[0])], fill=color, width=line_width + 1)
    for point in session.get('points', []):
        if point['band'] != band:
            continue
        rx, ry = point['reference']
        mx, my = point['moving']
        if point['status'] == 'confirmed':
            color = '#f4b860'
        else:
            color = '#9aafff'
        radius = 4
        draw.ellipse([rx - radius, ry - radius, rx + radius, ry + radius], outline=color, width=line_width)
        draw.line([rx, ry, mx, my], fill=color, width=1)
        draw.ellipse([mx - radius, my - radius, mx + radius, my + radius], outline='#ff7d7d', width=line_width)
        for candidate in point.get('candidates', []):
            cx, cy = candidate['moving']
            shade = '#ffd1d1' if candidate.get('ambiguous') else '#8fa5b7'
            draw.rectangle([cx - 2, cy - 2, cx + 2, cy + 2], outline=shade)
    maximum = c.integer(options.get('maximum', 1200), '预览边长', 64, 1600)
    image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()
