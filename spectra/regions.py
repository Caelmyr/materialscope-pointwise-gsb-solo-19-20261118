import copy

import numpy as np
from scipy import ndimage

from . import contracts as c
from .numerics import descriptive, histogram, deterministic_indices


def region_mask(cube, region):
    kind = c.choice(region.get('kind', 'rectangle'), '区域形状', ['rectangle', 'ellipse'])
    x0, y0, x1, y1 = c.bounds(region['bounds'], cube.width, cube.height)
    mask = np.zeros(cube.shape[:2], dtype=bool)
    if kind == 'rectangle':
        mask[y0:y1, x0:x1] = True
    else:
        yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
        xx = (xx + 0.5 - (x1 - x0) / 2) / ((x1 - x0) / 2)
        yy = (yy + 0.5 - (y1 - y0) / 2) / ((y1 - y0) / 2)
        mask[y0:y1, x0:x1] = xx ** 2 + yy ** 2 <= 1
    return mask


def validate_region(cube, asset_id, options):
    options = c.object_value(options, '分区')
    result = {
        'id': c.generated_id('region'),
        'asset_id': c.identifier(asset_id),
        'name': c.text(options.get('name', '取样区域'), '分区名称'),
        'kind': c.choice(options.get('kind', 'rectangle'), '区域形状', ['rectangle', 'ellipse']),
        'bounds': list(c.bounds(options.get('bounds'), cube.width, cube.height)),
        'color': c.color(options.get('color', '#f4b860')),
        'created_at': c.timestamp(),
    }
    if not region_mask(cube, result).any():
        raise c.Invalid('分区没有覆盖任何像素中心')
    return result


def spectrum_statistics(cube, mask):
    if mask.shape != cube.shape[:2]:
        raise c.Invalid('分区网格不一致')
    rows = []
    means = []
    spreads = []
    valid_bands = []
    for index, band in enumerate(cube.bands):
        valid = mask & cube.valid[:, :, index]
        values = cube.values[:, :, index]
        statistics = descriptive(values, valid)
        rows.append({'index': index, **band, **statistics})
        means.append(statistics['mean'])
        spreads.append(statistics['std'])
        valid_bands.append(statistics['count'] > 0)
    return {
        'bands': rows,
        'values': means,
        'std': spreads,
        'valid': valid_bands,
        'unit': cube.unit,
        'domain': cube.domain,
        'covered_pixels': int(mask.sum()),
        'common_valid_pixels': int((mask & cube.common_valid()).sum()),
        'area': float(mask.sum() * cube.grid['x_step'] * cube.grid['y_step']),
        'area_unit': cube.grid['unit'] + '^2',
    }


def region_report(cube, region):
    mask = region_mask(cube, region)
    result = spectrum_statistics(cube, mask)
    result['region'] = copy.deepcopy(region)
    result['fingerprint'] = cube.fingerprint()
    return result


def neighborhood_spectrum(cube, x, y, radius):
    x = c.integer(x, '横坐标', 0, cube.width - 1)
    y = c.integer(y, '纵坐标', 0, cube.height - 1)
    radius = c.integer(radius, '邻域半径', 0, 32)
    mask = np.zeros(cube.shape[:2], dtype=bool)
    x0, x1 = max(0, x - radius), min(cube.width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(cube.height, y + radius + 1)
    mask[y0:y1, x0:x1] = True
    report = spectrum_statistics(cube, mask)
    report['position'] = [x, y]
    report['radius'] = radius
    report['bounds'] = [x0, y0, x1, y1]
    return report


def correlation_report(cube, region=None):
    mask = cube.common_valid()
    if region is not None:
        mask &= region_mask(cube, region)
    indices = deterministic_indices(mask, 50000)
    if len(indices) < 3:
        raise c.Invalid('至少需要3个共同有效像素')
    samples = cube.values.reshape(-1, cube.count)[indices].astype(float)
    deviation = samples.std(axis=0)
    usable = deviation > 1e-12
    centered = samples - samples.mean(axis=0)
    normalized = np.divide(centered, deviation, out=np.zeros_like(centered), where=usable)
    matrix = normalized.T @ normalized / len(samples)
    return {
        'bands': [item['name'] for item in cube.bands],
        'matrix': matrix.tolist(),
        'constant_bands': np.flatnonzero(~usable).tolist(),
        'sample_count': len(indices),
        'available_count': int(mask.sum()),
        'sampling': 'deterministic up to 50000 complete spectra',
    }


def compare_regions(cube, regions):
    if len(regions) < 2:
        raise c.Invalid('至少需要两个分区')
    reports = [region_report(cube, region) for region in regions]
    comparisons = []
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            a = reports[i]
            b = reports[j]
            band_rows = []
            distances = []
            for index, band in enumerate(cube.bands):
                left = a['bands'][index]
                right = b['bands'][index]
                if not left['count'] or not right['count']:
                    band_rows.append({'band': band['name'], 'difference': None, 'standardized_difference': None})
                    continue
                delta = left['mean'] - right['mean']
                spread = np.sqrt((left['std'] ** 2 + right['std'] ** 2) / 2)
                effect = delta / spread if spread > 1e-12 else None
                band_rows.append({'band': band['name'], 'difference': delta, 'standardized_difference': effect})
                distances.append(delta ** 2)
            comparisons.append({
                'first_region': a['region']['id'],
                'second_region': b['region']['id'],
                'bands': band_rows,
                'mean_spectral_distance': float(np.sqrt(np.mean(distances))) if distances else None,
            })
    return {'regions': reports, 'comparisons': comparisons, 'unit': cube.unit}


def band_distribution(cube, band, region=None, bins=64):
    band = c.integer(band, '波段', 0, cube.count - 1)
    valid = cube.valid[:, :, band].copy()
    if region is not None:
        valid &= region_mask(cube, region)
    values = cube.values[:, :, band]
    return {
        'band': cube.bands[band],
        'statistics': descriptive(values, valid),
        'histogram': histogram(values, valid, bins),
        'unit': cube.unit,
    }


def line_profile(cube, start, end, band, samples=200):
    start = c.pair(start, '起点')
    end = c.pair(end, '终点')
    band = c.integer(band, '波段', 0, cube.count - 1)
    samples = c.integer(samples, '剖面采样数', 2, 1000)
    for x, y in [start, end]:
        if not 0 <= x < cube.width or not 0 <= y < cube.height:
            raise c.Invalid('剖面端点超出影像')
    x = np.linspace(start[0], end[0], samples)
    y = np.linspace(start[1], end[1], samples)
    ix = np.floor(x + 0.5).astype(int).clip(0, cube.width - 1)
    iy = np.floor(y + 0.5).astype(int).clip(0, cube.height - 1)
    valid = cube.valid[iy, ix, band]
    values = cube.values[iy, ix, band]
    distances = np.sqrt(((x - x[0]) * cube.grid['x_step']) ** 2 + ((y - y[0]) * cube.grid['y_step']) ** 2)
    return {
        'band': cube.bands[band],
        'points': [[float(px), float(py)] for px, py in zip(x, y)],
        'distance': distances.tolist(),
        'values': [float(v) if ok else None for v, ok in zip(values, valid)],
        'valid': valid.tolist(),
        'distance_unit': cube.grid['unit'],
        'value_unit': cube.unit,
        'sampling': 'nearest pixel center',
    }


def region_quality(cube, region):
    mask = region_mask(cube, region)
    total = int(mask.sum())
    rows = []
    for index, band in enumerate(cube.bands):
        valid = mask & cube.valid[:, :, index]
        field = cube.values[:, :, index].astype(float)
        horizontal = valid[:, 1:] & valid[:, :-1]
        vertical = valid[1:] & valid[:-1]
        hdiff = field[:, 1:] - field[:, :-1]
        vdiff = field[1:] - field[:-1]
        differences = np.concatenate([hdiff[horizontal], vdiff[vertical]])
        noise = float(np.median(np.abs(differences - np.median(differences))) / (0.67449 * np.sqrt(2))) if differences.size else None
        stats = descriptive(field, valid)
        rows.append({
            'band': band['name'],
            'coverage': int(valid.sum()) / max(total, 1),
            'local_difference_noise': noise,
            'signal_to_noise_proxy': abs(stats['mean']) / noise if stats['mean'] is not None and noise and noise > 0 else None,
            'statistics': stats,
        })
    return {'region': region, 'bands': rows, 'note': '相邻差分噪声代理也可能包含真实纹理，不是仪器噪声标定'}


def aggregate_region_table(cube, regions):
    output = []
    for region in regions:
        report = region_report(cube, region)
        for band in report['bands']:
            output.append({
                'region_id': region['id'],
                'region_name': region['name'],
                'band_name': band['name'],
                'wavelength_nm': band.get('wavelength_nm'),
                'count': band['count'],
                'mean': band['mean'],
                'std': band['std'],
                'minimum': band['min'],
                'maximum': band['max'],
                'unit': cube.unit,
                'covered_pixels': report['covered_pixels'],
            })
    return output
