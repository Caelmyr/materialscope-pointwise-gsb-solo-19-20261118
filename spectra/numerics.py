import numpy as np
from scipy import ndimage

from . import contracts as c


def descriptive(values, valid):
    selected = np.asarray(values, dtype=np.float64)[valid]
    if selected.size == 0:
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'std': None, 'median': None}
    return {
        'count': int(selected.size),
        'min': float(selected.min()),
        'max': float(selected.max()),
        'mean': float(selected.mean()),
        'std': float(selected.std()),
        'median': float(np.median(selected)),
        'percentiles': np.percentile(selected, [1, 5, 25, 75, 95, 99]).tolist(),
    }


def histogram(values, valid, bins=64):
    bins = c.integer(bins, '直方图桶数', 4, 256)
    selected = np.asarray(values)[valid]
    if not len(selected):
        return {'edges': [], 'counts': []}
    counts, edges = np.histogram(selected, bins=bins)
    return {'edges': edges.tolist(), 'counts': counts.tolist()}


def percentile_limits(values, valid, low=2, high=98):
    low, high = c.percent_range(low, high)
    selected = np.asarray(values)[valid]
    if not selected.size:
        return 0.0, 1.0
    a, b = np.percentile(selected, [low, high])
    if b <= a:
        b = a + max(abs(a) * 1e-6, 1e-6)
    return float(a), float(b)


def normalized_gaussian(values, valid, sigma, keep_missing=True):
    weights = ndimage.gaussian_filter(valid.astype(np.float64), sigma, mode='reflect')
    numerator = ndimage.gaussian_filter(np.where(valid, values, 0).astype(np.float64), sigma, mode='reflect')
    output = np.divide(numerator, weights, out=np.zeros_like(numerator), where=weights > 1e-10)
    output_valid = weights > 1e-6
    if keep_missing:
        output_valid &= valid
    return output, output_valid


def local_moments(values, valid, size):
    weights = ndimage.uniform_filter(valid.astype(np.float64), size=size, mode='reflect')
    first = ndimage.uniform_filter(np.where(valid, values, 0).astype(np.float64), size=size, mode='reflect')
    second = ndimage.uniform_filter(np.where(valid, values ** 2, 0).astype(np.float64), size=size, mode='reflect')
    mean = np.divide(first, weights, out=np.zeros_like(first), where=weights > 1e-12)
    square = np.divide(second, weights, out=np.zeros_like(second), where=weights > 1e-12)
    variance = np.maximum(square - mean ** 2, 0)
    return mean, variance, weights


def stable_covariance(samples, regularization=1e-5):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or len(samples) < max(3, samples.shape[1] + 1):
        raise c.Invalid('共同有效样本不足以估计协方差')
    center = samples.mean(axis=0)
    centered = samples - center
    covariance = centered.T @ centered / (len(samples) - 1)
    scale = max(float(np.trace(covariance) / len(center)), 1e-12)
    covariance += np.eye(len(center)) * regularization * scale
    return center, covariance


def deterministic_indices(mask, maximum, seed=71):
    indices = np.flatnonzero(mask)
    if len(indices) > maximum:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, maximum, replace=False))
    return indices


def chunk_ranges(total, chunk=20000):
    for start in range(0, total, chunk):
        yield start, min(start + chunk, total)


def stable_eigen(covariance):
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    return values, vectors


def safe_ratio(numerator, denominator, valid, epsilon=1e-10):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    valid = np.broadcast_to(valid, numerator.shape).copy()
    valid &= np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > epsilon)
    output = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64), where=valid)
    valid &= np.isfinite(output)
    output[~valid] = 0
    return output, valid


def masked_quantile(values, valid, axis, quantile):
    values = np.where(valid, values, np.nan)
    with np.errstate(invalid='ignore'):
        result = np.nanquantile(values, quantile, axis=axis)
    result_valid = np.isfinite(result)
    return np.where(result_valid, result, 0), result_valid


def robust_scale(values, axis=0):
    median = np.median(values, axis=axis)
    mad = np.median(np.abs(values - np.expand_dims(median, axis)), axis=axis)
    return median, np.maximum(mad * 1.4826, 1e-8)


def label_palette(count):
    indices = np.arange(count, dtype=float)
    hue = (indices * 0.6180339887498949) % 1
    saturation = np.full(count, 0.65)
    value = np.full(count, 0.95)
    h = hue * 6
    segment = np.floor(h).astype(int)
    fraction = h - segment
    p = value * (1 - saturation)
    q = value * (1 - fraction * saturation)
    t = value * (1 - (1 - fraction) * saturation)
    rgb = np.empty((count, 3), dtype=float)
    tables = [(value, t, p), (q, value, p), (p, value, t), (p, q, value), (t, p, value), (value, p, q)]
    for part, columns in enumerate(tables):
        selected = segment % 6 == part
        for channel in range(3):
            rgb[selected, channel] = columns[channel][selected]
    if count:
        rgb[0] = 0.1
    return np.round(rgb * 255).astype(np.uint8)


def normalized_vectors(values):
    lengths = np.linalg.norm(values, axis=-1)
    output = np.divide(values, lengths[..., None], out=np.zeros_like(values), where=lengths[..., None] > 1e-12)
    return output, lengths > 1e-12


def bilinear_sample(field, valid, x, y, strict=True):
    height, width = field.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    numerator = np.zeros_like(x, dtype=np.float64)
    denominator = np.zeros_like(x, dtype=np.float64)
    invalid_weight = np.zeros_like(x, dtype=np.float64)
    for dx, dy, weight in [
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ]:
        sx = x0 + dx
        sy = y0 + dy
        inside = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
        sx = np.clip(sx, 0, width - 1)
        sy = np.clip(sy, 0, height - 1)
        usable = inside & valid[sy, sx]
        numerator += np.where(usable, field[sy, sx] * weight, 0)
        denominator += np.where(usable, weight, 0)
        invalid_weight += np.where(usable, 0, weight)
    output_valid = denominator > 1e-8
    if strict:
        output_valid &= invalid_weight < 1e-7
    output = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=output_valid)
    return output, output_valid


def nearest_sample(field, valid, x, y):
    sx = np.floor(x + 0.5).astype(int)
    sy = np.floor(y + 0.5).astype(int)
    inside = (sx >= 0) & (sx < field.shape[1]) & (sy >= 0) & (sy < field.shape[0])
    sx = np.clip(sx, 0, field.shape[1] - 1)
    sy = np.clip(sy, 0, field.shape[0] - 1)
    output_valid = inside & valid[sy, sx]
    return np.where(output_valid, field[sy, sx], 0), output_valid
