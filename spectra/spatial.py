import numpy as np
from scipy import ndimage

from . import contracts as c
from .model import create_cube, source_preserving, derived_cube
from .numerics import normalized_gaussian, local_moments, bilinear_sample, nearest_sample, percentile_limits


def spatial_smooth(source, options, references):
    sigma = c.number(options.get('sigma', 1), '空间标准差', 0.1, 10)
    method = c.choice(options.get('method', 'gaussian'), '平滑方式', ['gaussian', 'median', 'edge_preserving'])
    size = c.odd_window(options.get('size', 3), maximum=15)
    strength = c.number(options.get('edge_strength', 0.1), '边缘保持强度', 0.00001, 100000)
    output = np.zeros_like(source.values)
    output_valid = source.valid.copy()
    for index in range(source.count):
        field = source.values[:, :, index].astype(float)
        valid = source.valid[:, :, index]
        if method == 'gaussian':
            result, usable = normalized_gaussian(field, valid, sigma)
        elif method == 'median':
            filled = np.where(valid, field, np.nan)
            result = ndimage.generic_filter(filled, np.nanmedian, size=size, mode='reflect')
            usable = valid & np.isfinite(result)
        else:
            mean, variance, weights = local_moments(field, valid, size)
            blend = variance / (variance + strength ** 2)
            result = mean + blend * (field - mean)
            usable = valid & (weights > 1e-8)
        output[:, :, index] = np.where(usable, result, 0)
        output_valid[:, :, index] = usable
    report = {'method': method, 'sigma': sigma, 'size': size, 'edge_strength': strength, 'missing_policy': 'excluded from neighborhoods and preserved at original pixels'}
    return source_preserving(source, output, output_valid, 'spatial_smooth', report), report


def spatial_gradient(source, options, references):
    band = c.integer(options.get('band', 0), '波段', 0, source.count - 1)
    sigma = c.number(options.get('sigma', 0), '预平滑标准差', 0, 10)
    field = source.values[:, :, band].astype(float)
    valid = source.valid[:, :, band]
    if sigma:
        field, valid = normalized_gaussian(field, valid, sigma)
    gx = ndimage.sobel(field, axis=1, mode='reflect') / (8 * source.grid['x_step'])
    gy = ndimage.sobel(field, axis=0, mode='reflect') / (8 * source.grid['y_step'])
    usable = ndimage.binary_erosion(valid, structure=np.ones((3, 3)), border_value=1)
    magnitude = np.hypot(gx, gy)
    direction = np.degrees(np.arctan2(gy, gx))
    output = np.stack([gx, gy, magnitude, direction], axis=2)
    report = {'band': band, 'sigma': sigma, 'spacing': source.grid, 'direction_unit': 'degrees', 'direction_meaning': 'clockwise in image x-right/y-down coordinates'}
    return derived_cube(source, output, usable, ['Gradient X', 'Gradient Y', 'Magnitude', 'Direction'], 'spatial_gradient', report, 'mixed'), report


def local_statistics(source, options, references):
    band = c.integer(options.get('band', 0), '波段', 0, source.count - 1)
    size = c.odd_window(options.get('size', 9), maximum=51)
    minimum_coverage = c.number(options.get('minimum_coverage', 0.5), '最小支持率', 0, 1)
    mean, variance, weights = local_moments(source.values[:, :, band].astype(float), source.valid[:, :, band], size)
    usable = source.valid[:, :, band] & (weights >= minimum_coverage)
    result = np.stack([mean, np.sqrt(variance), weights], axis=2)
    report = {'band': band, 'size': size, 'minimum_coverage': minimum_coverage, 'border': 'reflection', 'outputs': ['mean', 'standard deviation', 'observed support fraction']}
    return derived_cube(source, result, usable, ['Local mean', 'Local deviation', 'Support fraction'], 'local_statistics', report, 'mixed'), report


def texture_entropy(source, options, references):
    band = c.integer(options.get('band', 0), '波段', 0, source.count - 1)
    size = c.odd_window(options.get('size', 9), maximum=31)
    levels = c.integer(options.get('levels', 16), '量化级数', 4, 64)
    field = source.values[:, :, band].astype(float)
    valid = source.valid[:, :, band]
    low, high = percentile_limits(field, valid, 0, 100)
    quantized = np.clip(np.floor((field - low) / (high - low) * levels), 0, levels - 1).astype(int)
    denominator = ndimage.uniform_filter(valid.astype(float), size, mode='reflect')
    entropy = np.zeros(source.shape[:2], dtype=float)
    for level in range(levels):
        counts = ndimage.uniform_filter(((quantized == level) & valid).astype(float), size, mode='reflect')
        probability = np.divide(counts, denominator, out=np.zeros_like(counts), where=denominator > 1e-12)
        entropy -= probability * np.log2(np.maximum(probability, 1e-12))
    usable = valid & (denominator > 1e-12)
    report = {'band': band, 'size': size, 'levels': levels, 'range': [low, high], 'normalization': 'only observed neighborhood samples'}
    return derived_cube(source, entropy, usable, ['Local entropy'], 'texture_entropy', report, 'bit'), report


def texture_pairs(source, options, references):
    band = c.integer(options.get('band', 0), '波段', 0, source.count - 1)
    size = c.odd_window(options.get('size', 9), maximum=31)
    distance = c.integer(options.get('distance', 1), '像素间隔', 1, 10)
    direction = c.choice(options.get('direction', 'horizontal'), '方向', ['horizontal', 'vertical', 'diagonal'])
    levels = c.integer(options.get('levels', 16), '量化级数', 4, 64)
    field = source.values[:, :, band].astype(float)
    valid = source.valid[:, :, band]
    low, high = percentile_limits(field, valid, 0, 100)
    quantized = np.clip(np.floor((field - low) / (high - low) * levels), 0, levels - 1)
    dx = distance if direction != 'vertical' else 0
    dy = distance if direction != 'horizontal' else 0
    shifted = np.roll(quantized, (-dy, -dx), axis=(0, 1))
    paired = valid & np.roll(valid, (-dy, -dx), axis=(0, 1))
    if dx:
        paired[:, -dx:] = False
    if dy:
        paired[-dy:] = False
    delta = quantized - shifted
    features = [delta ** 2, np.abs(delta), 1 / (1 + delta ** 2)]
    denominator = ndimage.uniform_filter(paired.astype(float), size, mode='constant')
    result = []
    for feature in features:
        numerator = ndimage.uniform_filter(np.where(paired, feature, 0), size, mode='constant')
        result.append(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12))
    usable = valid & (denominator > 1e-12)
    report = {'band': band, 'size': size, 'direction': direction, 'distance': distance, 'levels': levels, 'quantization_range': [low, high], 'pair_count': int(paired.sum())}
    return derived_cube(source, np.stack(result, axis=2), usable, ['Contrast', 'Dissimilarity', 'Homogeneity'], 'texture_pairs', report), report


def crop_cube(source, options, references):
    x0, y0, x1, y1 = c.bounds(options.get('bounds'), source.width, source.height)
    grid = dict(source.grid)
    grid['origin'] = [source.grid['origin'][0] + x0 * source.grid['x_step'], source.grid['origin'][1] + y0 * source.grid['y_step']]
    report = {'bounds': [x0, y0, x1, y1], 'origin_pixel_offset': [x0, y0]}
    metadata = {'operation': 'crop', 'source_fingerprint': source.fingerprint(), 'report': report}
    output = create_cube(source.values[y0:y1, x0:x1], source.valid[y0:y1, x0:x1], list(source.bands), source.domain, source.unit, grid, metadata)
    return output, report


def resize_cube(source, options, references):
    width = c.integer(options.get('width', source.width), '宽度', 1, 1024)
    height = c.integer(options.get('height', source.height), '高度', 1, 1024)
    method = c.choice(options.get('method', 'bilinear'), '采样', ['bilinear', 'nearest'])
    if source.domain == 'labels' and method != 'nearest':
        raise c.Invalid('标签图只能使用最近邻，避免制造不存在的标签')
    yy, xx = np.indices((height, width), dtype=float)
    xx = (xx + 0.5) * source.width / width - 0.5
    yy = (yy + 0.5) * source.height / height - 0.5
    xx = np.clip(xx, 0, source.width - 1)
    yy = np.clip(yy, 0, source.height - 1)
    output = np.zeros((height, width, source.count), dtype=np.float32)
    valid = np.zeros_like(output, dtype=bool)
    for index in range(source.count):
        if method == 'nearest':
            result, mask = nearest_sample(source.values[:, :, index], source.valid[:, :, index], xx, yy)
        else:
            result, mask = bilinear_sample(source.values[:, :, index], source.valid[:, :, index], xx, yy, strict=True)
        output[:, :, index] = result
        valid[:, :, index] = mask
    grid = dict(source.grid)
    grid['x_step'] *= source.width / width
    grid['y_step'] *= source.height / height
    report = {'width': width, 'height': height, 'method': method, 'extent_preserved': True, 'missing_policy': 'strict contributing support'}
    metadata = {'operation': 'resize', 'source_fingerprint': source.fingerprint(), 'report': report}
    return create_cube(output, valid, list(source.bands), source.domain, source.unit, grid, metadata), report


def select_subset(source, options, references):
    bands = c.index_list(options.get('bands'), source.count)
    report = {'bands': bands, 'names': [source.bands[index]['name'] for index in bands]}
    metadata = {'operation': 'band_subset', 'source_fingerprint': source.fingerprint(), 'report': report}
    return create_cube(source.values[:, :, bands], source.valid[:, :, bands], [source.bands[index] for index in bands], source.domain, source.unit, source.grid, metadata), report


def mask_morphology(source, options, references):
    mode = c.choice(options.get('mode', 'erode'), '有效域操作', ['erode', 'remove_small_islands'])
    radius = c.integer(options.get('radius', 1), '收缩半径', 1, 20)
    minimum = c.integer(options.get('minimum_pixels', 16), '最小连通面积', 1, source.width * source.height)
    valid = source.valid.copy()
    for band in range(source.count):
        if mode == 'erode':
            valid[:, :, band] = ndimage.binary_erosion(valid[:, :, band], iterations=radius, border_value=0)
        else:
            labels, _ = ndimage.label(valid[:, :, band])
            counts = np.bincount(labels.ravel())
            keep = counts >= minimum
            keep[0] = False
            valid[:, :, band] = keep[labels]
    report = {'mode': mode, 'radius': radius, 'minimum_pixels': minimum, 'removed_samples': int(source.valid.sum() - valid.sum())}
    return source_preserving(source, source.values, valid, 'mask_morphology', report), report


def background_plane(source, options, references):
    band = c.integer(options.get('band', 0), '拟合波段', 0, source.count - 1)
    border_width = c.integer(options.get('border_width', 10), '边缘宽度', 1, max(1, min(source.width, source.height) // 3))
    robust_iterations = c.integer(options.get('robust_iterations', 4), '稳健迭代次数', 0, 10)
    yy, xx = np.indices(source.shape[:2], dtype=float)
    nx = (xx + 0.5) / source.width * 2 - 1
    ny = (yy + 0.5) / source.height * 2 - 1
    border = (xx < border_width) | (yy < border_width) | (xx >= source.width - border_width) | (yy >= source.height - border_width)
    selected = border & source.valid[:, :, band]
    if selected.sum() < 20:
        raise c.Invalid('边缘有效样本不足')
    design = np.column_stack([np.ones(int(selected.sum())), nx[selected], ny[selected]])
    values = source.values[:, :, band][selected].astype(float)
    weights = np.ones(len(values))
    coefficients = np.zeros(3)
    for _ in range(robust_iterations + 1):
        coefficients = np.linalg.lstsq(design * np.sqrt(weights[:, None]), values * np.sqrt(weights), rcond=None)[0]
        residual = values - design @ coefficients
        scale = max(float(np.median(np.abs(residual - np.median(residual))) * 1.4826), 1e-9)
        weights = np.minimum(1, 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
    plane = coefficients[0] + coefficients[1] * nx + coefficients[2] * ny
    output = source.values[:, :, band].astype(float) - plane
    report = {'band': band, 'border_width': border_width, 'coefficients': coefficients.tolist(), 'fit_samples': int(selected.sum()), 'robust_iterations': robust_iterations, 'note': '边缘被视为背景；真实边缘材料梯度也会被减去'}
    return derived_cube(source, output, source.valid[:, :, band], ['Background removed'], 'background_plane', report, source.unit), report
