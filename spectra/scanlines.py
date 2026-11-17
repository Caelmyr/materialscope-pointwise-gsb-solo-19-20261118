import numpy as np
from scipy import ndimage

from . import contracts as c
from .model import source_preserving, derived_cube
from .numerics import masked_quantile, robust_scale


def scanline_statistics(source, options):
    direction = c.choice(options.get('direction', 'columns'), '扫描方向', ['columns', 'rows'])
    band = c.integer(options.get('band', 0), '波段', 0, source.count - 1)
    minimum = c.integer(options.get('minimum_support', 8), '最少支持像素', 1, max(source.width, source.height))
    axis = 0 if direction == 'columns' else 1
    field = source.values[:, :, band].astype(float)
    valid = source.valid[:, :, band]
    count = valid.sum(axis=axis)
    median, usable = masked_quantile(field, valid, axis, 0.5)
    lower, _ = masked_quantile(field, valid, axis, 0.25)
    upper, _ = masked_quantile(field, valid, axis, 0.75)
    usable &= count >= minimum
    global_median = float(np.median(field[valid])) if valid.any() else None
    line_center, line_scale = robust_scale(median[usable]) if usable.any() else (0.0, 1.0)
    standardized = (median - line_center) / line_scale
    return {
        'direction': direction,
        'band': band,
        'name': source.bands[band]['name'],
        'minimum_support': minimum,
        'count': count.tolist(),
        'usable': usable.tolist(),
        'median': [float(item) if ok else None for item, ok in zip(median, usable)],
        'q25': [float(item) if ok else None for item, ok in zip(lower, usable)],
        'q75': [float(item) if ok else None for item, ok in zip(upper, usable)],
        'median_standardized': [float(item) if ok else None for item, ok in zip(standardized, usable)],
        'global_median': global_median,
        'interpretation': '扫描线差异可能来自真实材料结构；校正前必须确认均匀参考区域',
    }


def validate_reference_rectangle(source, options):
    if 'reference_bounds' not in options:
        raise c.Invalid('条带校正必须指定均匀参考矩形')
    x0, y0, x1, y1 = c.bounds(options['reference_bounds'], source.width, source.height)
    direction = c.choice(options.get('direction', 'columns'), '方向', ['columns', 'rows'])
    if direction == 'columns' and (x0 != 0 or x1 != source.width):
        raise c.Invalid('列条带参考区域必须横跨整个影像宽度')
    if direction == 'rows' and (y0 != 0 or y1 != source.height):
        raise c.Invalid('行条带参考区域必须纵跨整个影像高度')
    return (x0, y0, x1, y1), direction


def remove_stripes(source, options, references):
    bounds, direction = validate_reference_rectangle(source, options)
    x0, y0, x1, y1 = bounds
    minimum = c.integer(options.get('minimum_support', 8), '最少支持像素', 2, max(source.width, source.height))
    method = c.choice(options.get('method', 'offset'), '条带模型', ['offset', 'gain'])
    smooth_width = c.odd_window(options.get('trend_window', 15), '趋势窗口', 101)
    maximum_correction = c.number(options.get('maximum_correction', 0.2), '最大校正幅度', 0.000001, 1e9)
    axis = 0 if direction == 'columns' else 1
    output = source.values.copy()
    valid_output = source.valid.copy()
    rows = []
    for band in range(source.count):
        field = source.values[y0:y1, x0:x1, band].astype(float)
        valid = source.valid[y0:y1, x0:x1, band]
        median, usable = masked_quantile(field, valid, axis, 0.5)
        count = valid.sum(axis=axis)
        usable &= count >= minimum
        if usable.sum() < 3:
            raise c.Invalid('波段' + source.bands[band]['name'] + '的参考扫描线不足')
        indices = np.arange(len(median))
        interpolated = np.interp(indices, indices[usable], median[usable])
        trend = ndimage.median_filter(interpolated, size=smooth_width, mode='nearest')
        if method == 'offset':
            correction = np.clip(interpolated - trend, -maximum_correction, maximum_correction)
            correction[~usable] = 0
            applied = correction[None, :] if direction == 'columns' else correction[:, None]
            result = source.values[:, :, band] - applied
        else:
            ratio = np.divide(trend, interpolated, out=np.ones_like(trend), where=np.abs(interpolated) > 1e-10)
            ratio = np.clip(ratio, max(0, 1 - maximum_correction), 1 + maximum_correction)
            ratio[~usable] = 1
            applied = ratio[None, :] if direction == 'columns' else ratio[:, None]
            correction = ratio
            result = source.values[:, :, band] * applied
        output[:, :, band] = result
        rows.append({
            'band': source.bands[band]['name'],
            'supported_lines': int(usable.sum()),
            'unsupported_lines': np.flatnonzero(~usable).tolist(),
            'reference_median': median.tolist(),
            'smooth_trend': trend.tolist(),
            'applied_correction': correction.tolist(),
        })
    report = {
        'direction': direction,
        'reference_bounds': list(bounds),
        'method': method,
        'trend_window': smooth_width,
        'maximum_correction': maximum_correction,
        'bands': rows,
        'unsupported_policy': 'leave unchanged',
        'assumption': 'reference rectangle has no true high-frequency scanline material variation',
    }
    return source_preserving(source, output, valid_output, 'remove_stripes', report), report


def frequency_diagnostic(source, options):
    band = c.integer(options.get('band', 0), '波段', 0, source.count - 1)
    direction = c.choice(options.get('direction', 'columns'), '方向', ['columns', 'rows'])
    axis = 0 if direction == 'columns' else 1
    field = source.values[:, :, band]
    valid = source.valid[:, :, band]
    count = valid.sum(axis=axis)
    signal = np.divide(np.sum(np.where(valid, field, 0), axis=axis), count, out=np.zeros_like(count, dtype=float), where=count > 0)
    usable = count > 0
    if usable.sum() < 8:
        raise c.Invalid('有效扫描线不足8条')
    coordinates = np.arange(len(signal))
    filled = np.interp(coordinates, coordinates[usable], signal[usable])
    design = np.column_stack([np.ones(len(filled)), np.linspace(-1, 1, len(filled))])
    trend = design @ np.linalg.lstsq(design, filled, rcond=None)[0]
    windowed = (filled - trend) * np.hanning(len(filled))
    power = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(len(filled))
    power[0] = 0
    peaks = np.argsort(power)[::-1][:min(8, max(0, len(power) - 1))]
    spacing = source.grid['x_step'] if direction == 'columns' else source.grid['y_step']
    return {
        'band': band,
        'direction': direction,
        'frequency_cycles_per_pixel': frequencies.tolist(),
        'power': power.tolist(),
        'peaks': [{'frequency': float(frequencies[index]), 'period_pixels': float(1 / frequencies[index]) if frequencies[index] else None, 'period_physical': float(spacing / frequencies[index]) if frequencies[index] else None, 'power': float(power[index])} for index in peaks],
        'length_unit': source.grid['unit'],
        'missing_lines_interpolated_for_diagnostic': np.flatnonzero(~usable).tolist(),
        'note': '线性去趋势与Hann窗后频谱；峰值也可能是试片纹理，不自动判为仪器条带',
    }


def spectral_noise_map(source, options, references):
    wavelengths, order = source.sorted_wavelengths()
    if source.count < 5:
        raise c.Invalid('局部谱残差诊断至少5波段')
    first = source.values[:, :, order[:-2]].astype(float)
    center = source.values[:, :, order[1:-1]].astype(float)
    last = source.values[:, :, order[2:]].astype(float)
    left = wavelengths[1:-1] - wavelengths[:-2]
    right = wavelengths[2:] - wavelengths[1:-1]
    prediction = (first * right + last * left) / (left + right)
    residual = center - prediction
    valid = source.valid[:, :, order[:-2]] & source.valid[:, :, order[1:-1]] & source.valid[:, :, order[2:]]
    minimum = c.integer(options.get('minimum_triplets', 3), '最少有效三带组', 1, source.count - 2)
    count = valid.sum(axis=2)
    squared = np.sum(np.where(valid, residual ** 2, 0), axis=2)
    rms = np.sqrt(np.divide(squared, count, out=np.zeros(source.shape[:2]), where=count > 0))
    usable = count >= minimum
    report = {
        'minimum_triplets': minimum,
        'wavelengths_nm': wavelengths.tolist(),
        'valid_triplet_count_distribution': np.bincount(count.ravel(), minlength=source.count - 1).tolist(),
        'interpretation': '三带线性预测残差含真实吸收结构，不能直接视为纯噪声方差',
    }
    return derived_cube(source, rms, usable, ['Spectral local residual'], 'spectral_noise_map', report, source.unit), report
