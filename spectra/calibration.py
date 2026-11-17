import numpy as np

from . import contracts as c
from .model import create_cube, same_spectral_grid, same_spatial_grid
from .numerics import descriptive


def validate_reference(source, reference, label):
    same_spectral_grid(source, reference)
    same_spatial_grid(source, reference)
    if reference.domain != 'raw' or source.domain != 'raw':
        raise c.Invalid(label + '与样品均须为原始数值域')
    if reference.unit != source.unit:
        raise c.Invalid(label + '与样品原始单位不一致')


def calibrate(source, options, references):
    dark = references.get('dark')
    white = references.get('white')
    if dark is None or white is None:
        raise c.Invalid('请指定同网格暗场和白参考影像')
    validate_reference(source, dark, '暗场')
    validate_reference(source, white, '白参考')
    white_reflectance = c.number(options.get('white_reflectance', 0.99), '白板反射率', 0.01, 1)
    minimum_span = c.number(options.get('minimum_span', 1), '最低白暗差', 1e-8, 1e9)
    saturation = options.get('saturation')
    if saturation is not None:
        saturation = c.number(saturation, '饱和值', 0.01, 1e12)
    numerator = source.values.astype(np.float64) - dark.values
    denominator = white.values.astype(np.float64) - dark.values
    valid = source.valid & dark.valid & white.valid
    valid &= denominator > minimum_span
    if saturation is not None:
        valid &= (source.values < saturation) & (white.values < saturation)
    result = np.divide(numerator * white_reflectance, denominator, out=np.zeros_like(numerator), where=valid)
    below = int((valid & (result < 0)).sum())
    above = int((valid & (result > 1)).sum())
    clipping = c.choice(options.get('range_policy', 'preserve'), '范围处理', ['preserve', 'mask', 'clip'])
    if clipping == 'mask':
        valid &= (result >= 0) & (result <= 1)
    elif clipping == 'clip':
        result = np.clip(result, 0, 1)
    report = {
        'white_reflectance': white_reflectance,
        'minimum_span': minimum_span,
        'saturation': saturation,
        'range_policy': clipping,
        'below_zero_count': below,
        'above_one_count': above,
        'source_valid_count': int(source.valid.sum()),
        'calibrated_valid_count': int(valid.sum()),
        'dark_fingerprint': dark.fingerprint(),
        'white_fingerprint': white.fingerprint(),
        'formula': '(sample-dark)/(white-dark)*white_reflectance',
        'band_reports': [],
    }
    for index, band in enumerate(source.bands):
        report['band_reports'].append({
            'band': band['name'],
            'span': descriptive(denominator[:, :, index], dark.valid[:, :, index] & white.valid[:, :, index]),
            'reflectance': descriptive(result[:, :, index], valid[:, :, index]),
            'invalidated': int(source.valid[:, :, index].sum() - valid[:, :, index].sum()),
        })
    metadata = {
        'operation': 'calibrate',
        'source_fingerprint': source.fingerprint(),
        'report': report,
    }
    cube = create_cube(result, valid, list(source.bands), 'reflectance', 'fraction', source.grid, metadata)
    return cube, report


def linear_adjustment(source, options, references):
    count = source.count
    gains = c.array_value(options.get('gains', [1] * count), '增益', count, count)
    offsets = c.array_value(options.get('offsets', [0] * count), '偏移', count, count)
    gains = np.asarray([c.number(item, '增益', -1e6, 1e6) for item in gains])
    offsets = np.asarray([c.number(item, '偏移', -1e12, 1e12) for item in offsets])
    output = source.values.astype(np.float64) * gains + offsets
    report = {'gains': gains.tolist(), 'offsets': offsets.tolist(), 'note': '用户显式数值修订；不自动解释为反射率标定'}
    domain = c.choice(options.get('domain', source.domain), '数值域', ['raw', 'reflectance', 'derived'])
    unit = c.text(options.get('unit', source.unit), '单位', 32)
    cube = create_cube(output, source.valid, list(source.bands), domain, unit, source.grid, {
        'operation': 'linear_adjustment',
        'source_fingerprint': source.fingerprint(),
        'report': report,
    })
    return cube, report


def mask_range(source, options, references):
    bands = c.index_list(options.get('bands', list(range(source.count))), source.count)
    low = c.number(options.get('low', 0), '下界', -1e12, 1e12)
    high = c.number(options.get('high', 1), '上界', -1e12, 1e12)
    if low >= high:
        raise c.Invalid('下界必须小于上界')
    propagation = c.choice(options.get('propagation', 'per_band'), '传播方式', ['per_band', 'all_bands'])
    valid = source.valid.copy()
    selected_values = source.values[:, :, bands]
    in_range = (selected_values >= low) & (selected_values <= high)
    if propagation == 'per_band':
        valid[:, :, bands] &= in_range
    else:
        keep = np.all(in_range & source.valid[:, :, bands], axis=2)
        valid &= keep[:, :, None]
    report = {
        'bands': bands,
        'low': low,
        'high': high,
        'propagation': propagation,
        'removed_samples': int(source.valid.sum() - valid.sum()),
    }
    cube = create_cube(source.values, valid, list(source.bands), source.domain, source.unit, source.grid, {
        'operation': 'mask_range',
        'source_fingerprint': source.fingerprint(),
        'report': report,
    })
    return cube, report
