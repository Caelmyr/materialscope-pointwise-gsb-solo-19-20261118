import numpy as np
from scipy import ndimage

from . import contracts as c
from .model import source_preserving
from .numerics import bilinear_sample


def phase_translation(reference, moving, valid, max_shift):
    if valid.sum() < 64:
        raise c.Invalid('配准共同有效区域不足64像素')
    height, width = reference.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    a = np.where(valid, reference - np.mean(reference[valid]), 0) * window
    b = np.where(valid, moving - np.mean(moving[valid]), 0) * window
    if np.linalg.norm(a) < 1e-10 or np.linalg.norm(b) < 1e-10:
        raise c.Invalid('配准影像缺少可辨识结构')
    cross = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.ifft2(cross).real
    sy = np.fft.fftfreq(height) * height
    sx = np.fft.fftfreq(width) * width
    allowed = (np.abs(sy[:, None]) <= max_shift) & (np.abs(sx[None, :]) <= max_shift)
    restricted = np.where(allowed, correlation, -np.inf)
    py, px = np.unravel_index(np.argmax(restricted), restricted.shape)
    dy = float(sy[py])
    dx = float(sx[px])
    def correction(before, peak, after):
        denominator = before - 2 * peak + after
        if abs(denominator) < 1e-10:
            return 0.0
        return float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))
    dy += correction(correlation[(py - 1) % height, px], correlation[py, px], correlation[(py + 1) % height, px])
    dx += correction(correlation[py, (px - 1) % width], correlation[py, px], correlation[py, (px + 1) % width])
    neighborhood = correlation.copy()
    for oy in range(-2, 3):
        for ox in range(-2, 3):
            neighborhood[(py + oy) % height, (px + ox) % width] = 0
    side = neighborhood[allowed]
    score = (correlation[py, px] - np.mean(side)) / max(float(np.std(side)), 1e-12)
    return dx, dy, float(score)


def register_bands(source, options, references):
    reference_band = c.integer(options.get('reference_band', source.count // 2), '参考波段', 0, source.count - 1)
    max_shift = c.integer(options.get('max_shift', 12), '最大平移', 1, min(100, max(1, min(source.height, source.width) // 4)))
    minimum_score = c.number(options.get('minimum_score', 3), '最低相关峰质量', 0, 1000)
    reject = c.choice(options.get('uncertain_policy', 'keep'), '低置信处理', ['keep', 'reject'])
    yy, xx = np.indices(source.shape[:2], dtype=float)
    result = np.zeros_like(source.values)
    output_valid = np.zeros_like(source.valid)
    reports = []
    reference = source.values[:, :, reference_band].astype(float)
    for index, band in enumerate(source.bands):
        if index == reference_band:
            result[:, :, index] = source.values[:, :, index]
            output_valid[:, :, index] = source.valid[:, :, index]
            reports.append({'band': band['name'], 'dx': 0, 'dy': 0, 'status': 'reference', 'score': None})
            continue
        common = source.valid[:, :, reference_band] & source.valid[:, :, index]
        try:
            dx, dy, score = phase_translation(reference, source.values[:, :, index], common, max_shift)
            accepted = score >= minimum_score and abs(dx) <= max_shift and abs(dy) <= max_shift
            reason = 'peak_confidence' if not accepted else None
        except c.Invalid as error:
            dx, dy, score = 0.0, 0.0, 0.0
            accepted = False
            reason = str(error)
        if not accepted:
            if reject == 'reject':
                raise c.Invalid('波段' + band['name'] + '配准不确定：' + str(reason))
            dx = dy = 0.0
        values, valid = bilinear_sample(source.values[:, :, index], source.valid[:, :, index], xx - dx, yy - dy, strict=True)
        result[:, :, index] = values
        output_valid[:, :, index] = valid
        aligned_common = output_valid[:, :, index] & source.valid[:, :, reference_band]
        before_error = float(np.mean(np.abs(source.values[:, :, index][common] - reference[common]))) if common.any() else None
        after_error = float(np.mean(np.abs(values[aligned_common] - reference[aligned_common]))) if aligned_common.any() else None
        reports.append({
            'band': band['name'],
            'dx': dx,
            'dy': dy,
            'score': score,
            'status': 'registered' if accepted else 'unchanged_uncertain',
            'reason': reason,
            'before_absolute_difference': before_error,
            'after_absolute_difference': after_error,
            'valid_count': int(valid.sum()),
        })
    report = {'reference_band': reference_band, 'model': 'translation', 'uncertain_policy': reject, 'bands': reports}
    return source_preserving(source, result, output_valid, 'register_bands', report), report


def manual_band_shift(source, options, references):
    shifts = c.array_value(options.get('shifts'), '逐带平移', source.count, source.count)
    shifts = [c.pair(item, '平移像素', -100, 100) for item in shifts]
    yy, xx = np.indices(source.shape[:2], dtype=float)
    values = np.zeros_like(source.values)
    valid = np.zeros_like(source.valid)
    for index, (dx, dy) in enumerate(shifts):
        values[:, :, index], valid[:, :, index] = bilinear_sample(
            source.values[:, :, index], source.valid[:, :, index], xx - dx, yy - dy, strict=True,
        )
    report = {'shifts': shifts, 'meaning': 'positive dx/dy moves image content right/down', 'sampling': 'strict bilinear support'}
    return source_preserving(source, values, valid, 'manual_band_shift', report), report
