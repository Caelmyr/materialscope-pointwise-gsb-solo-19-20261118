import numpy as np

from . import contracts as c
from .model import create_cube, same_spectral_grid, same_spatial_grid
from .numerics import descriptive


def validate_repeated(cubes):
    if not 2 <= len(cubes) <= 12:
        raise c.Invalid('重复测量需要2–12幅影像')
    first = cubes[0]
    for cube in cubes[1:]:
        same_spectral_grid(first, cube)
        same_spatial_grid(first, cube)
        if cube.domain != first.domain or cube.unit != first.unit:
            raise c.Invalid('重复测量的数值域与单位必须一致')
    total = sum(cube.values.nbytes + cube.valid.nbytes for cube in cubes)
    if total > 180000000:
        raise c.Invalid('重复测量工作集超过180MB，请先裁切')
    return first


def combine_repeated(cubes, options):
    first = validate_repeated(cubes)
    method = c.choice(options.get('method', 'mean'), '重复测量汇总', ['mean', 'median', 'sigma_clip'])
    minimum = c.integer(options.get('minimum_observations', 2), '最少观测次数', 1, len(cubes))
    sigma = c.number(options.get('sigma', 3), '剔除阈值标准差', 1, 10)
    values = np.stack([cube.values for cube in cubes], axis=0).astype(float)
    valid = np.stack([cube.valid for cube in cubes], axis=0)
    original_count = valid.sum(axis=0)
    if method == 'sigma_clip':
        median = np.nanmedian(np.where(valid, values, np.nan), axis=0)
        mad = np.nanmedian(np.abs(np.where(valid, values - median, np.nan)), axis=0)
        spread = np.maximum(mad * 1.4826, 1e-8)
        valid &= np.abs(values - median) <= sigma * spread
    count = valid.sum(axis=0)
    if method == 'median':
        output = np.nanmedian(np.where(valid, values, np.nan), axis=0)
    else:
        output = np.divide(np.sum(np.where(valid, values, 0), axis=0), count, out=np.zeros(first.shape), where=count > 0)
    usable = count >= minimum
    deviation = np.where(valid, values - output, 0)
    variance = np.divide(np.sum(deviation ** 2, axis=0), count - 1, out=np.zeros(first.shape), where=count > 1)
    standard_error = np.sqrt(np.divide(variance, count, out=np.zeros(first.shape), where=count > 0))
    report = {
        'method': method,
        'minimum_observations': minimum,
        'sigma': sigma,
        'source_fingerprints': [cube.fingerprint() for cube in cubes],
        'source_count': len(cubes),
        'rejected_samples': int(original_count.sum() - count.sum()),
        'observation_count_distribution': np.bincount(count.ravel(), minlength=len(cubes) + 1).tolist(),
        'bands': [],
    }
    for band in range(first.count):
        report['bands'].append({
            'band': first.bands[band]['name'],
            'repeat_deviation': descriptive(np.sqrt(variance[:, :, band]), usable[:, :, band]),
            'standard_error': descriptive(standard_error[:, :, band], usable[:, :, band]),
        })
    combined = create_cube(output, usable, list(first.bands), first.domain, first.unit, first.grid, {'operation': 'repeat_combine', 'report': report, 'source_fingerprint': first.fingerprint()})
    uncertainty = create_cube(standard_error, usable & (count > 1), list(first.bands), 'derived', first.unit, first.grid, {'operation': 'repeat_standard_error', 'report': report, 'source_fingerprint': combined.fingerprint()})
    return combined, uncertainty, report


def repeat_stability(cubes):
    first = validate_repeated(cubes)
    output = []
    for index, cube in enumerate(cubes):
        rows = []
        for band in range(first.count):
            common = first.valid[:, :, band] & cube.valid[:, :, band]
            a = first.values[:, :, band][common].astype(float)
            b = cube.values[:, :, band][common].astype(float)
            if len(a) < 3:
                rows.append({'band': first.bands[band]['name'], 'count': len(a), 'status': 'insufficient'})
                continue
            design = np.column_stack([a, np.ones(len(a))])
            scale, offset = np.linalg.lstsq(design, b, rcond=None)[0]
            residual = b - (a * scale + offset)
            rows.append({
                'band': first.bands[band]['name'],
                'count': len(a),
                'status': 'measured',
                'gain_relative_to_first': float(scale),
                'offset_relative_to_first': float(offset),
                'residual_rms': float(np.sqrt(np.mean(residual ** 2))),
                'mean_difference': float(np.mean(b - a)),
            })
        output.append({'measurement_index': index, 'fingerprint': cube.fingerprint(), 'bands': rows})
    return {'measurements': output, 'reference_index': 0, 'note': '同位置重复测量的漂移诊断，不将真实材料变化自动解释为仪器漂移'}
