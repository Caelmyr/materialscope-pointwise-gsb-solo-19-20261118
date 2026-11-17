import numpy as np

from . import contracts as c
from .model import derived_cube
from .numerics import normalized_vectors, chunk_ranges
from .regions import region_mask, spectrum_statistics


def signature_from_region(cube, region, name):
    if cube.domain != 'reflectance':
        raise c.Invalid('材料谱库仅接受反射率域光谱')
    waves = cube.wavelengths()
    report = spectrum_statistics(cube, region_mask(cube, region))
    valid = np.asarray(report['valid'])
    if valid.sum() < 3:
        raise c.Invalid('材料参考至少需要3个有效波段')
    order = np.argsort(waves[valid])
    selected = np.flatnonzero(valid)[order]
    return {
        'id': c.generated_id('signature'),
        'name': c.text(name, '材料名称'),
        'wavelengths_nm': waves[selected].tolist(),
        'values': [report['values'][index] for index in selected],
        'std': [report['std'][index] for index in selected],
        'unit': cube.unit,
        'domain': cube.domain,
        'source_fingerprint': cube.fingerprint(),
        'source_region': region['id'],
        'source_counts': [report['bands'][index]['count'] for index in selected],
        'created_at': c.timestamp(),
    }


def validate_signature(options):
    options = c.object_value(options, '材料参考')
    waves = c.finite_array(options.get('wavelengths_nm'), '波长', 1, 512)
    values = c.finite_array(options.get('values'), '反射率', 1, 512)
    if len(waves) != len(values) or len(waves) < 3:
        raise c.Invalid('材料参考需至少3组对应的波长与反射率')
    if np.any(waves < 100) or np.any(waves > 30000) or len(set(waves)) != len(waves):
        raise c.Invalid('波长超界或重复')
    if np.any(values < -0.1) or np.any(values > 2):
        raise c.Invalid('参考反射率须在-0.1至2之间')
    order = np.argsort(waves)
    return {
        'id': c.generated_id('signature'),
        'name': c.text(options.get('name', '外部材料'), '材料名称'),
        'wavelengths_nm': waves[order].tolist(),
        'values': values[order].tolist(),
        'std': None,
        'unit': 'fraction',
        'domain': 'reflectance',
        'source_fingerprint': None,
        'source_region': None,
        'source_counts': None,
        'created_at': c.timestamp(),
    }


def resample_signature(signature, wavelengths, maximum_gap):
    source_waves = np.asarray(signature['wavelengths_nm'], dtype=float)
    source_values = np.asarray(signature['values'], dtype=float)
    order = np.argsort(source_waves)
    source_waves = source_waves[order]
    source_values = source_values[order]
    output = np.interp(wavelengths, source_waves, source_values)
    valid = (wavelengths >= source_waves[0]) & (wavelengths <= source_waves[-1])
    right = np.searchsorted(source_waves, wavelengths, side='left')
    right = np.clip(right, 0, len(source_waves) - 1)
    left = np.maximum(right - 1, 0)
    exact = np.abs(source_waves[right] - wavelengths) < 1e-7
    gaps = source_waves[right] - source_waves[left]
    valid &= exact | (gaps <= maximum_gap)
    return output, valid


def library_matrix(cube, signatures, maximum_gap=1000):
    waves = cube.wavelengths()
    if cube.domain != 'reflectance' or cube.unit != 'fraction':
        raise c.Invalid('谱库搜索需要反射率fraction域')
    if not 1 <= len(signatures) <= 100:
        raise c.Invalid('请选择1–100条材料参考')
    values = []
    valid = []
    for signature in signatures:
        if signature.get('domain') != cube.domain or signature.get('unit') != cube.unit:
            raise c.Invalid('参考光谱数值域或单位不一致')
        row, mask = resample_signature(signature, waves, maximum_gap)
        values.append(row)
        valid.append(mask)
    return np.asarray(values), np.asarray(valid)


def match_library(source, options, references):
    signatures = references.get('signatures', [])
    maximum_gap = c.number(options.get('maximum_gap_nm', 1000), '最大插值跨距', 1, 30000)
    matrix, library_valid = library_matrix(source, signatures, maximum_gap)
    common_bands = np.all(library_valid, axis=0)
    if common_bands.sum() < 3:
        raise c.Invalid('全部参考共同覆盖的波段不足3个')
    bands = np.flatnonzero(common_bands)
    image_valid = source.common_valid(bands)
    selected = np.flatnonzero(image_valid)
    if not len(selected):
        raise c.Invalid('没有共同有效的完整光谱')
    method = c.choice(options.get('method', 'angle'), '相似度', ['angle', 'rmse'])
    maximum_score = c.number(options.get('maximum_score', 0.2), '最大匹配距离', 0, 1e8)
    minimum_margin = c.number(options.get('minimum_margin', 0), '最小领先距离', 0, 1e8)
    references_selected = matrix[:, bands]
    if method == 'angle':
        references_selected, ref_usable = normalized_vectors(references_selected)
        if not ref_usable.all():
            raise c.Invalid('存在零长度参考光谱')
    flat = source.values[:, :, bands].reshape(-1, len(bands))
    labels = np.zeros(source.width * source.height, dtype=np.float32)
    best_distance = np.zeros_like(labels)
    margins = np.zeros_like(labels)
    valid_flat = image_valid.ravel().copy()
    for start, end in chunk_ranges(len(selected)):
        indices = selected[start:end]
        samples = flat[indices].astype(float)
        if method == 'angle':
            samples, sample_usable = normalized_vectors(samples)
            scores = np.arccos(np.clip(samples @ references_selected.T, -1, 1))
            valid_flat[indices] &= sample_usable
        else:
            scores = np.sqrt(np.maximum(np.mean((samples[:, None, :] - references_selected[None, :, :]) ** 2, axis=2), 0))
        order = np.argsort(scores, axis=1, kind='stable')
        first = scores[np.arange(len(indices)), order[:, 0]]
        second = scores[np.arange(len(indices)), order[:, 1]] if len(signatures) > 1 else np.full(len(indices), np.inf)
        margin = second - first
        accepted = (first <= maximum_score) & (margin >= minimum_margin) & valid_flat[indices]
        labels[indices] = np.where(accepted, order[:, 0] + 1, 0)
        best_distance[indices] = first
        margins[indices] = np.where(np.isfinite(margin), margin, 0)
    label_field = labels.reshape(source.shape[:2])
    report = {
        'method': method,
        'score_unit': 'radian' if method == 'angle' else 'reflectance',
        'common_bands': bands.tolist(),
        'maximum_score': maximum_score,
        'minimum_margin': minimum_margin,
        'labels': [{'label': i + 1, 'signature_id': item['id'], 'name': item['name'], 'pixels': int((labels == i + 1).sum())} for i, item in enumerate(signatures)],
        'unmatched_pixels': int(((labels == 0) & valid_flat).sum()),
        'matching_policy': 'all references use the same common spectral support; zero is unmatched',
        'best_distance_mean': float(best_distance[valid_flat].mean()),
    }
    cube = derived_cube(source, label_field, valid_flat.reshape(source.shape[:2]), ['Material match'], 'match_library', report, unit='class_id', domain='labels')
    return cube, report


def signature_comparison(signatures):
    if not 2 <= len(signatures) <= 20:
        raise c.Invalid('请选择2–20条材料参考')
    lower = max(min(item['wavelengths_nm']) for item in signatures)
    upper = min(max(item['wavelengths_nm']) for item in signatures)
    if lower >= upper:
        raise c.Invalid('参考光谱没有共同波长范围')
    wavelengths = np.unique(np.concatenate([np.asarray(item['wavelengths_nm']) for item in signatures]))
    wavelengths = wavelengths[(wavelengths >= lower) & (wavelengths <= upper)]
    sampled = [resample_signature(item, wavelengths, 30000)[0] for item in signatures]
    normalized, valid = normalized_vectors(np.asarray(sampled))
    angles = np.arccos(np.clip(normalized @ normalized.T, -1, 1))
    errors = np.sqrt(np.mean((np.asarray(sampled)[:, None, :] - np.asarray(sampled)[None, :, :]) ** 2, axis=2))
    return {
        'names': [item['name'] for item in signatures],
        'wavelengths_nm': wavelengths.tolist(),
        'values': np.asarray(sampled).tolist(),
        'angle_matrix': angles.tolist(),
        'rmse_matrix': errors.tolist(),
        'nonzero': valid.tolist(),
        'unit': 'fraction',
    }
