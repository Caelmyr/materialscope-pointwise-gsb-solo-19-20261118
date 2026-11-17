import numpy as np
from scipy import ndimage

from . import contracts as c
from .model import derived_cube
from .numerics import descriptive


def otsu_threshold(values):
    if values.size < 2:
        raise c.Invalid('阈值分割样本不足')
    if float(np.max(values)) == float(np.min(values)):
        return float(values[0])
    counts, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    weights = counts.astype(float) / counts.sum()
    cumulative = np.cumsum(weights)
    cumulative_mean = np.cumsum(weights * centers)
    global_mean = cumulative_mean[-1]
    denominator = cumulative * (1 - cumulative)
    variance = np.divide((global_mean * cumulative - cumulative_mean) ** 2, denominator, out=np.zeros_like(denominator), where=denominator > 1e-12)
    return float(centers[int(np.argmax(variance))])


def object_measurements(labels, valid, grid, intensities=None):
    objects = ndimage.find_objects(labels)
    rows = []
    pixel_area = grid['x_step'] * grid['y_step']
    for label, slices in enumerate(objects, 1):
        if slices is None:
            continue
        local = labels[slices] == label
        yy, xx = np.nonzero(local)
        if len(xx) == 0:
            continue
        x = xx + slices[1].start
        y = yy + slices[0].start
        cx = float(np.mean(x) + 0.5)
        cy = float(np.mean(y) + 0.5)
        coordinates = np.column_stack([(x + 0.5 - cx) * grid['x_step'], (y + 0.5 - cy) * grid['y_step']])
        covariance = coordinates.T @ coordinates / len(x)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = int(np.argmax(eigenvalues))
        orientation = float(np.degrees(np.arctan2(eigenvectors[1, major], eigenvectors[0, major])))
        major_length = float(4 * np.sqrt(max(eigenvalues[major], 0)))
        minor_length = float(4 * np.sqrt(max(eigenvalues[1 - major], 0)))
        padded = np.pad(local, 1, constant_values=False)
        horizontal_edges = np.sum(padded[:, 1:] != padded[:, :-1])
        vertical_edges = np.sum(padded[1:] != padded[:-1])
        perimeter = float(horizontal_edges * grid['y_step'] + vertical_edges * grid['x_step'])
        border = bool((x == 0).any() or (x == labels.shape[1] - 1).any() or (y == 0).any() or (y == labels.shape[0] - 1).any())
        touches_missing = bool(ndimage.binary_dilation(labels == label, iterations=1)[~valid].any())
        row = {
            'label': label,
            'pixels': int(len(x)),
            'area': float(len(x) * pixel_area),
            'area_unit': grid['unit'] + '^2',
            'centroid_pixel': [cx, cy],
            'centroid_world': [grid['origin'][0] + cx * grid['x_step'], grid['origin'][1] + cy * grid['y_step']],
            'bounds': [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)],
            'major_length': major_length,
            'minor_length': minor_length,
            'orientation_degrees': orientation,
            'pixel_edge_perimeter': perimeter,
            'circularity': float(4 * np.pi * len(x) * pixel_area / perimeter ** 2) if perimeter else 0,
            'touches_image_border': border,
            'touches_missing': touches_missing,
        }
        if intensities is not None:
            selected = intensities[y, x]
            row['intensity'] = descriptive(selected, np.ones(len(selected), dtype=bool))
        rows.append(row)
    return rows


def segment_objects(source, options, references):
    band = c.integer(options.get('band', 0), '分割波段', 0, source.count - 1)
    field = source.values[:, :, band]
    valid = source.valid[:, :, band]
    method = c.choice(options.get('method', 'otsu'), '阈值方式', ['otsu', 'manual', 'percentile'])
    if method == 'otsu':
        threshold = otsu_threshold(field[valid])
    elif method == 'percentile':
        quantile = c.number(options.get('percentile', 75), '分位', 0, 100)
        threshold = float(np.percentile(field[valid], quantile))
    else:
        threshold = c.number(options.get('threshold', 0.5), '阈值', -1e12, 1e12)
    direction = c.choice(options.get('direction', 'above'), '保留方向', ['above', 'below'])
    foreground = (field > threshold if direction == 'above' else field < threshold) & valid
    opening = c.integer(options.get('opening', 0), '开运算半径', 0, 5)
    closing = c.integer(options.get('closing', 0), '闭运算半径', 0, 5)
    if opening:
        foreground = ndimage.binary_opening(foreground, iterations=opening) & valid
    if closing:
        foreground = ndimage.binary_closing(foreground, iterations=closing) & valid
    connectivity = c.choice(options.get('connectivity', 8), '连通性', [4, 8])
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, count = ndimage.label(foreground, structure)
    minimum_area = c.integer(options.get('minimum_pixels', 10), '最小像素面积', 1, source.width * source.height)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_area
    keep[0] = False
    labels[~keep[labels]] = 0
    old_ids = np.flatnonzero(keep)
    remap = np.zeros(len(keep), dtype=int)
    remap[old_ids] = np.arange(1, len(old_ids) + 1)
    labels = remap[labels]
    objects = object_measurements(labels, valid, source.grid, field)
    report = {
        'band': band,
        'method': method,
        'threshold': threshold,
        'direction': direction,
        'connectivity': connectivity,
        'minimum_pixels': minimum_area,
        'opening': opening,
        'closing': closing,
        'raw_component_count': count,
        'objects': objects,
        'object_count': len(objects),
        'measurement_semantics': 'pixel-edge perimeter; covariance ellipse; invalid pixels are excluded',
    }
    return derived_cube(source, labels, valid, ['Object labels'], 'segment_objects', report, 'object_id', domain='labels'), report


def label_filter(source, options, references):
    if source.domain != 'labels' or source.count != 1:
        raise c.Invalid('对象筛选需要单波段标签图')
    minimum = c.integer(options.get('minimum_pixels', 10), '最小面积', 1, source.width * source.height)
    maximum = c.integer(options.get('maximum_pixels', source.width * source.height), '最大面积', minimum, source.width * source.height)
    exclude_border = c.boolean(options.get('exclude_border', False), '去除边界对象')
    labels = source.values[:, :, 0].astype(np.int32)
    valid = source.valid[:, :, 0]
    rows = object_measurements(labels, valid, source.grid)
    kept = []
    reasons = []
    for row in rows:
        reason = []
        if not minimum <= row['pixels'] <= maximum:
            reason.append('area')
        if exclude_border and (row['touches_image_border'] or row['touches_missing']):
            reason.append('boundary')
        if not reason:
            kept.append(row['label'])
        else:
            reasons.append({'label': row['label'], 'reasons': reason})
    output = np.where(np.isin(labels, kept), labels, 0)
    report = {'kept_labels': kept, 'removed': reasons, 'objects': [row for row in rows if row['label'] in kept], 'identity_policy': 'original label IDs preserved, no renumbering'}
    return derived_cube(source, output, valid, ['Filtered labels'], 'label_filter', report, 'object_id', domain='labels'), report


def label_distance(source, options, references):
    if source.domain != 'labels' or source.count != 1:
        raise c.Invalid('标签距离需要单波段标签图')
    labels = c.array_value(options.get('labels', [1]), '目标标签', 1, 100)
    labels = [c.integer(item, '标签', 1, 1000000) for item in labels]
    field = source.values[:, :, 0].astype(int)
    valid = source.valid[:, :, 0]
    targets = np.isin(field, labels) & valid
    if not targets.any():
        raise c.Invalid('所选标签不存在')
    distance = ndimage.distance_transform_edt(~targets, sampling=(source.grid['y_step'], source.grid['x_step']))
    report = {'labels': labels, 'target_pixels': int(targets.sum()), 'metric': 'Euclidean center distance; missing pixels are not obstacles'}
    return derived_cube(source, distance, valid, ['Distance to labels'], 'label_distance', report, source.grid['unit']), report
