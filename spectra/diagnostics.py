import numpy as np
from scipy import ndimage

from . import contracts as c
from .numerics import descriptive, histogram, robust_scale, masked_quantile, deterministic_indices
from .regions import region_mask


def validity_report(cube):
    counts = cube.valid.sum(axis=(0, 1))
    per_pixel = cube.valid.sum(axis=2)
    distribution = np.bincount(per_pixel.ravel(), minlength=cube.count + 1)
    all_valid = cube.common_valid()
    any_valid = cube.any_valid()
    islands, island_count = ndimage.label(any_valid)
    island_sizes = np.bincount(islands.ravel())[1:]
    holes, hole_count = ndimage.label(~any_valid)
    border_ids = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
    hole_sizes = np.bincount(holes.ravel())
    internal_holes = [int(hole_sizes[index]) for index in range(1, len(hole_sizes)) if index not in border_ids]
    bands = []
    for index, band in enumerate(cube.bands):
        mask = cube.valid[:, :, index]
        row_coverage = mask.mean(axis=1)
        column_coverage = mask.mean(axis=0)
        bands.append({
            'index': index,
            'name': band['name'],
            'valid_count': int(counts[index]),
            'fraction': float(counts[index] / (cube.width * cube.height)),
            'empty_rows': np.flatnonzero(row_coverage == 0).tolist(),
            'empty_columns': np.flatnonzero(column_coverage == 0).tolist(),
            'row_coverage': row_coverage.tolist(),
            'column_coverage': column_coverage.tolist(),
        })
    return {
        'bands': bands,
        'valid_band_count_distribution': distribution.tolist(),
        'all_valid_pixels': int(all_valid.sum()),
        'any_valid_pixels': int(any_valid.sum()),
        'partial_pixels': int((any_valid & ~all_valid).sum()),
        'spatial_components': island_count,
        'largest_component_pixels': int(island_sizes.max()) if len(island_sizes) else 0,
        'internal_missing_hole_sizes': internal_holes,
        'missing_meaning': 'unobserved or deliberately excluded; stored zeros do not represent measurements',
    }


def spectral_spacing_report(cube):
    waves = cube.wavelengths(required=False)
    present = np.isfinite(waves)
    order = np.argsort(waves[present])
    sorted_waves = waves[present][order]
    gaps = np.diff(sorted_waves)
    bands = np.flatnonzero(present)[order]
    return {
        'complete_metadata': bool(present.all()),
        'missing_band_indices': np.flatnonzero(~present).tolist(),
        'sorted_indices': bands.tolist(),
        'wavelengths_nm': sorted_waves.tolist(),
        'gaps_nm': gaps.tolist(),
        'minimum_gap_nm': float(gaps.min()) if len(gaps) else None,
        'maximum_gap_nm': float(gaps.max()) if len(gaps) else None,
        'median_gap_nm': float(np.median(gaps)) if len(gaps) else None,
        'uniform_sampling': bool(len(gaps) and np.allclose(gaps, gaps[0], rtol=0.001, atol=1e-6)),
        'extent_nm': [float(sorted_waves[0]), float(sorted_waves[-1])] if len(sorted_waves) else [],
        'note': '中心波长不是完整仪器响应曲线，带宽仅作记录',
    }


def band_quality_report(cube, band):
    band = c.integer(band, '波段', 0, cube.count - 1)
    values = cube.values[:, :, band].astype(float)
    valid = cube.valid[:, :, band]
    stats = descriptive(values, valid)
    selected = values[valid]
    if not len(selected):
        return {'band': cube.bands[band], 'statistics': stats, 'warnings': ['no_valid_samples']}
    low = float(np.min(selected))
    high = float(np.max(selected))
    same_low = int((selected == low).sum())
    same_high = int((selected == high).sum())
    row_medians, row_valid = masked_quantile(values, valid, 1, 0.5)
    col_medians, col_valid = masked_quantile(values, valid, 0, 0.5)
    horizontal_pairs = valid[:, 1:] & valid[:, :-1]
    vertical_pairs = valid[1:] & valid[:-1]
    horizontal_difference = (values[:, 1:] - values[:, :-1])[horizontal_pairs]
    vertical_difference = (values[1:] - values[:-1])[vertical_pairs]
    noise_h = float(robust_scale(horizontal_difference)[1] / np.sqrt(2)) if len(horizontal_difference) else None
    noise_v = float(robust_scale(vertical_difference)[1] / np.sqrt(2)) if len(vertical_difference) else None
    warnings = []
    if stats['std'] == 0:
        warnings.append('constant_band')
    if same_high / len(selected) > 0.1:
        warnings.append('many_samples_at_maximum_check_saturation')
    if valid.mean() < 0.5:
        warnings.append('less_than_half_observed')
    return {
        'band': cube.bands[band],
        'statistics': stats,
        'histogram': histogram(values, valid),
        'minimum_ties': same_low,
        'maximum_ties': same_high,
        'row_medians': [float(value) if ok else None for value, ok in zip(row_medians, row_valid)],
        'column_medians': [float(value) if ok else None for value, ok in zip(col_medians, col_valid)],
        'horizontal_noise_proxy': noise_h,
        'vertical_noise_proxy': noise_v,
        'warnings': warnings,
        'noise_note': '相邻差分包含材料纹理，不能代替暗场噪声测量',
    }


def reference_uniformity(cube, region=None):
    if cube.domain not in ('raw', 'reflectance'):
        raise c.Invalid('参考均匀性需要原始或反射率数据')
    spatial = np.ones(cube.shape[:2], dtype=bool) if region is None else region_mask(cube, region)
    yy, xx = np.indices(cube.shape[:2], dtype=float)
    nx = (xx + 0.5) / cube.width * 2 - 1
    ny = (yy + 0.5) / cube.height * 2 - 1
    rows = []
    for band in range(cube.count):
        mask = spatial & cube.valid[:, :, band]
        indices = deterministic_indices(mask, 50000)
        if len(indices) < 10:
            rows.append({'band': cube.bands[band]['name'], 'count': len(indices), 'status': 'insufficient'})
            continue
        design = np.column_stack([np.ones(len(indices)), nx.ravel()[indices], ny.ravel()[indices]])
        values = cube.values[:, :, band].ravel()[indices].astype(float)
        fit = np.linalg.lstsq(design, values, rcond=None)[0]
        residual = values - design @ fit
        center = float(np.mean(values))
        rows.append({
            'band': cube.bands[band]['name'],
            'count': len(indices),
            'status': 'measured',
            'mean': center,
            'coefficient_of_variation': float(np.std(values) / abs(center)) if abs(center) > 1e-12 else None,
            'horizontal_gradient': float(fit[1]),
            'vertical_gradient': float(fit[2]),
            'plane_residual_rms': float(np.sqrt(np.mean(residual ** 2))),
            'relative_plane_residual_rms': float(np.sqrt(np.mean(residual ** 2)) / abs(center)) if abs(center) > 1e-12 else None,
        })
    return {'bands': rows, 'region_id': region['id'] if region else None, 'note': '均匀性诊断不会修改像素；材料场景本身的空间变化也会进入统计'}


def metadata_audit(cube):
    warnings = []
    if any(band['wavelength_nm'] is None for band in cube.bands):
        warnings.append('部分波段缺少波长，物理波长处理不可用')
    if cube.domain == 'reflectance' and cube.unit != 'fraction':
        warnings.append('反射率域单位不是fraction，材料谱库搜索不可用')
    if cube.domain == 'labels' and cube.count != 1:
        warnings.append('多波段标签图不适合单对象操作')
    if cube.domain == 'labels':
        values = cube.values[cube.valid]
        if np.any(values < 0) or np.any(values != np.floor(values)):
            warnings.append('标签域含负数或非整数')
    if cube.grid['unit'] == 'pixel':
        warnings.append('未设置物理像素尺寸，面积只以像素平方报告')
    calibration = cube.metadata.get('operation') == 'calibrate'
    if cube.domain == 'reflectance' and not calibration and 'source_fingerprint' not in cube.metadata:
        warnings.append('导入数据自声明反射率，应用没有自动验证仪器标定')
    return {
        'domain': cube.domain,
        'unit': cube.unit,
        'grid': cube.grid,
        'source_operation': cube.metadata.get('operation', 'import'),
        'has_parent_fingerprint': 'source_fingerprint' in cube.metadata,
        'warnings': warnings,
        'fingerprint': cube.fingerprint(),
    }


def pairwise_band_difference(cube, first, second):
    first = c.integer(first, '第一波段', 0, cube.count - 1)
    second = c.integer(second, '第二波段', 0, cube.count - 1)
    valid = cube.valid[:, :, first] & cube.valid[:, :, second]
    if valid.sum() < 3:
        raise c.Invalid('共同有效样本不足')
    a = cube.values[:, :, first][valid].astype(float)
    b = cube.values[:, :, second][valid].astype(float)
    difference = a - b
    denominator = np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean())
    correlation = float(np.dot(a - a.mean(), b - b.mean()) / denominator) if denominator > 1e-12 else None
    indices = np.linspace(0, len(a) - 1, min(1000, len(a))).astype(int)
    return {
        'first': cube.bands[first],
        'second': cube.bands[second],
        'common_samples': len(a),
        'bias': float(difference.mean()),
        'rmse': float(np.sqrt(np.mean(difference ** 2))),
        'correlation': correlation,
        'difference_percentiles': np.percentile(difference, [1, 5, 50, 95, 99]).tolist(),
        'scatter': np.column_stack([a[indices], b[indices]]).tolist(),
        'interpretation': 'different wavelengths may legitimately differ; this is not a registration accuracy score',
    }


def processing_capabilities(cube):
    complete = cube.common_valid()
    wave_count = sum(item['wavelength_nm'] is not None for item in cube.bands)
    checks = {
        'pca': cube.count >= 2 and complete.sum() >= max(3, cube.count + 1),
        'cluster': complete.sum() >= 15,
        'anomaly': complete.sum() >= max(3, cube.count + 1),
        'spectral_smooth': wave_count == cube.count,
        'spectral_derivative': wave_count == cube.count and cube.count >= 3,
        'continuum_remove': wave_count == cube.count and cube.count >= 3 and complete.sum() <= 100000,
        'match_library': wave_count == cube.count and cube.domain == 'reflectance' and cube.unit == 'fraction',
        'calibrate': cube.domain == 'raw',
        'label_filter': cube.domain == 'labels' and cube.count == 1,
        'pca_reconstruct': cube.metadata.get('operation') == 'pca',
    }
    return {
        'checks': checks,
        'complete_spectra': int(complete.sum()),
        'band_count': cube.count,
        'physical_wavelength_count': wave_count,
        'limits': c.LIMITS,
        'note': '这里只给输入层预检，实际执行仍检查全部参数和参考影像',
    }
