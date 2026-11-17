from . import contracts as c
from . import calibration, registration, spectral, multivariate, segmentation, spatial, library, scanlines


OPERATIONS = {
    'remove_stripes': ('均匀参考条带校正', '标定与几何', scanlines.remove_stripes, {'direction': 'columns', 'reference_bounds': [0, 10, 192, 30], 'method': 'offset', 'trend_window': 15, 'maximum_correction': 0.2}),
    'spectral_noise_map': ('局部光谱残差图', '材料分析', scanlines.spectral_noise_map, {'minimum_triplets': 3}),
    'calibrate': ('暗白参考标定', '标定与几何', calibration.calibrate, {'white_reflectance': 0.99, 'minimum_span': 1, 'range_policy': 'preserve'}),
    'linear_adjustment': ('逐带增益偏移', '标定与几何', calibration.linear_adjustment, {}),
    'mask_range': ('数值范围有效域', '标定与几何', calibration.mask_range, {'low': 0, 'high': 1, 'propagation': 'per_band'}),
    'register_bands': ('波段平移配准', '标定与几何', registration.register_bands, {'reference_band': 5, 'max_shift': 12, 'minimum_score': 3, 'uncertain_policy': 'keep'}),
    'manual_band_shift': ('逐带手动平移', '标定与几何', registration.manual_band_shift, {}),
    'crop': ('裁切试片', '标定与几何', spatial.crop_cube, {'bounds': [10, 10, 120, 100]}),
    'resize': ('保持范围缩放', '标定与几何', spatial.resize_cube, {'width': 128, 'height': 96, 'method': 'bilinear'}),
    'band_subset': ('波段子集', '光谱处理', spatial.select_subset, {'bands': [0, 2, 4, 6, 8, 10]}),
    'spectral_smooth': ('物理波长平滑', '光谱处理', spectral.smooth_spectra, {'half_width_nm': 100, 'minimum_support': 2, 'preserve_missing': True}),
    'spectral_derivative': ('不等间隔波长导数', '光谱处理', spectral.spectral_derivative, {'order': 1}),
    'spectral_normalize': ('光谱归一化', '光谱处理', spectral.normalize_spectra, {'method': 'snv', 'minimum_bands': 3}),
    'continuum_remove': ('连续谱去除', '光谱处理', spectral.continuum_remove, {'output': 'ratio'}),
    'continuum_features': ('吸收特征图', '光谱处理', spectral.continuum_remove, {'output': 'features'}),
    'band_formula': ('受限波段表达式', '光谱处理', spectral.band_formula, {'expression': '(b8-b3)/(b8+b3)', 'name': 'Contrast index'}),
    'band_ratio': ('双波段指数', '光谱处理', spectral.band_ratio, {'first': 8, 'second': 3, 'mode': 'normalized_difference'}),
    'pca': ('主成分分解', '材料分析', multivariate.principal_components, {'components': 3, 'standardize': False}),
    'noise_components': ('噪声调整分量', '材料分析', multivariate.noise_adjusted_components, {'components': 3, 'regularization': 0.001}),
    'pca_reconstruct': ('主成分低秩重建', '材料分析', multivariate.reconstruct_components, {'components': 2}),
    'cluster': ('材料光谱聚类', '材料分析', multivariate.cluster_materials, {'clusters': 4, 'iterations': 50, 'standardize': True}),
    'anomaly': ('全局光谱异常', '材料分析', multivariate.anomaly_detection, {'regularization': 0.001, 'flag_percentile': 99}),
    'match_library': ('材料参考谱搜索', '材料分析', library.match_library, {'method': 'angle', 'maximum_score': 0.2, 'minimum_margin': 0.01}),
    'segment_objects': ('阈值对象分割', '对象与纹理', segmentation.segment_objects, {'band': 0, 'method': 'otsu', 'minimum_pixels': 10, 'connectivity': 8}),
    'label_filter': ('对象面积与边界筛选', '对象与纹理', segmentation.label_filter, {'minimum_pixels': 20, 'exclude_border': False}),
    'label_distance': ('到标签的物理距离', '对象与纹理', segmentation.label_distance, {'labels': [1]}),
    'spatial_smooth': ('有效域空间平滑', '对象与纹理', spatial.spatial_smooth, {'method': 'gaussian', 'sigma': 1, 'size': 3}),
    'spatial_gradient': ('物理空间梯度', '对象与纹理', spatial.spatial_gradient, {'band': 0, 'sigma': 1}),
    'local_statistics': ('邻域均值与离散', '对象与纹理', spatial.local_statistics, {'band': 0, 'size': 9, 'minimum_coverage': 0.5}),
    'texture_entropy': ('局部量化熵', '对象与纹理', spatial.texture_entropy, {'band': 0, 'size': 9, 'levels': 16}),
    'texture_pairs': ('方向邻接纹理', '对象与纹理', spatial.texture_pairs, {'band': 0, 'size': 9, 'direction': 'horizontal'}),
    'mask_morphology': ('有效域边缘收缩', '对象与纹理', spatial.mask_morphology, {'mode': 'erode', 'radius': 1}),
    'background_plane': ('稳健背景平面去除', '对象与纹理', spatial.background_plane, {'band': 0, 'border_width': 10, 'robust_iterations': 4}),
}


def operation_catalog():
    return [{'id': key, 'name': item[0], 'group': item[1], 'defaults': item[3]} for key, item in OPERATIONS.items()]


def run_operation(source, operation, options, references=None):
    if operation not in OPERATIONS:
        raise c.Invalid('未知处理操作')
    options = c.options_copy(options)
    references = references or {}
    name, group, function, defaults = OPERATIONS[operation]
    merged = {**defaults, **options}
    if operation == 'continuum_features':
        merged['output'] = 'features'
    if operation == 'register_bands' and 'reference_band' not in options:
        merged['reference_band'] = source.count // 2
    if operation == 'band_ratio' and 'first' not in options:
        merged['first'] = source.count - 1
        merged['second'] = max(0, source.count // 3)
    result, report = function(source, merged, references)
    report['operation_id'] = operation
    report['operation_name'] = name
    report['input_shape'] = list(source.shape)
    report['output_shape'] = list(result.shape)
    report['input_domain'] = source.domain
    report['output_domain'] = result.domain
    report['source_fingerprint'] = source.fingerprint()
    report['result_fingerprint'] = result.fingerprint()
    return result, c.json_safe(report), merged
