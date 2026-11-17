import numpy as np

from . import contracts as c
from .engine import OPERATIONS
from .model import same_spatial_grid, same_spectral_grid


def preflight_operation(cube, operation, options, project):
    if operation not in OPERATIONS:
        raise c.Invalid('未知操作')
    options = c.options_copy(options)
    issues = []
    warnings = []
    requirements = []
    shape = list(cube.shape)
    output_count = cube.count
    wave_complete = all(item['wavelength_nm'] is not None for item in cube.bands)
    complete = cube.common_valid()
    complete_count = int(complete.sum())
    spectral_ops = {'spectral_smooth', 'spectral_derivative', 'continuum_remove', 'continuum_features', 'match_library'}
    if operation in spectral_ops:
        requirements.append('每波段具有唯一物理波长')
        if not wave_complete:
            issues.append('缺少波长元数据')
    if operation in ('pca', 'noise_components'):
        requirements.append('至少两个波段及足够共同有效光谱')
        if cube.count < 2 or complete_count < max(3, cube.count + 1):
            issues.append('波段或共同有效样本不足')
        output_count = int(options.get('components', min(3, cube.count)))
        warnings.append('拟合最多使用50000个固定种子完整光谱；缺测谱不用于拟合或投影')
    if operation == 'anomaly':
        if complete_count < max(3, cube.count + 1):
            issues.append('协方差拟合样本不足')
        warnings.append('全局背景异常距离不是缺陷概率，稀有正常材料也可能高分')
        output_count = 1
    if operation == 'cluster':
        clusters = int(options.get('clusters', 5))
        if complete_count < clusters * 3:
            issues.append('聚类完整样本不足')
        warnings.append('簇标号没有预定义材料含义；使用最多30000个样本训练')
        output_count = 1
    if operation == 'calibrate':
        requirements.extend(['原始数值域', '同网格同波长暗场和白参考'])
        if cube.domain != 'raw':
            issues.append('输入不是原始数值域')
        assets = {item['id']: item for item in project['assets']}
        for role in ('dark', 'white'):
            selected = options.get(role + '_id')
            if not selected or selected not in assets:
                issues.append('缺少' + role + '参考')
        warnings.append('白暗差过小和参考无效样本将使对应波段无效')
    if operation == 'match_library':
        requirements.append('fraction反射率域及材料参考')
        if cube.domain != 'reflectance' or cube.unit != 'fraction':
            issues.append('输入不是fraction反射率')
        if not project['signatures']:
            issues.append('工程材料谱库为空')
        warnings.append('所有候选共用相同光谱覆盖，标签0表示未匹配')
        output_count = 1
    if operation in ('continuum_remove', 'continuum_features'):
        if cube.count < 3:
            issues.append('至少需要3个物理波长')
        if complete_count > 100000:
            issues.append('完整光谱超过10万，需先裁切')
        warnings.append('上包络基于实测采样点，不是连续真实吸收谱的恢复')
        output_count = 3 if operation == 'continuum_features' else cube.count
    if operation == 'spectral_derivative':
        if cube.count < 3:
            issues.append('至少需要3个波段')
        output_count = max(cube.count - 2, 1)
        warnings.append('输出忽略两端波段，并要求三相邻波长均有效')
    if operation in ('label_filter', 'label_distance'):
        if cube.domain != 'labels' or cube.count != 1:
            issues.append('需要单波段标签图')
        output_count = 1
    if operation == 'pca_reconstruct':
        if cube.metadata.get('operation') != 'pca':
            issues.append('输入不是带基底元数据的PCA结果')
        output_count = len(cube.metadata.get('report', {}).get('bands', [])) or cube.count
        warnings.append('低秩重建保留派生域，不自动宣称恢复原始反射率')
    if operation in ('band_formula', 'band_ratio', 'segment_objects', 'texture_entropy', 'background_plane'):
        output_count = 1
    if operation in ('local_statistics', 'texture_pairs'):
        output_count = 3
    if operation == 'spatial_gradient':
        output_count = 4
    if operation in ('register_bands', 'manual_band_shift'):
        warnings.append('严格插值支持会收缩有效域；不同波长的结构差异可能使自动平移不确定')
    if operation == 'resize':
        shape[0] = c.integer(options.get('height', cube.height), '高度', 1, 1024)
        shape[1] = c.integer(options.get('width', cube.width), '宽度', 1, 1024)
        if cube.domain == 'labels' and options.get('method', 'bilinear') != 'nearest':
            issues.append('标签图只能最近邻缩放')
    if operation == 'crop' and 'bounds' in options:
        x0, y0, x1, y1 = c.bounds(options['bounds'], cube.width, cube.height)
        shape[0] = y1 - y0
        shape[1] = x1 - x0
    if operation == 'band_subset':
        output_count = len(c.index_list(options.get('bands'), cube.count))
    shape[2] = output_count
    if np.prod(shape) > c.LIMITS['samples']:
        issues.append('预计结果超过1600万采样值')
    if cube.domain == 'labels' and operation in ('spectral_smooth', 'spectral_derivative', 'pca', 'anomaly', 'cluster'):
        warnings.append('标签编号是类别而非连续测量，对其数值分析通常没有材料意义')
    return {
        'operation': operation,
        'name': OPERATIONS[operation][0],
        'can_start': not issues,
        'issues': issues,
        'warnings': warnings,
        'requirements': requirements,
        'input_shape': list(cube.shape),
        'expected_output_shape': shape,
        'minimum_output_array_bytes': int(np.prod(shape) * 5),
        'complete_spectrum_count': complete_count,
        'estimate_scope': '结果数组下界，不含算法临时内存；不作速度承诺',
    }


def recipe_preflight(cube, steps, project):
    steps = c.array_value(steps, '配方步骤', 1, 6)
    reports = []
    for index, step in enumerate(steps):
        operation = c.text(step.get('operation'), '操作')
        if operation not in OPERATIONS:
            raise c.Invalid('配方包含未知操作')
        if index == 0:
            reports.append(preflight_operation(cube, operation, step.get('options', {}), project))
        else:
            reports.append({
                'operation': operation,
                'name': OPERATIONS[operation][0],
                'can_start': None,
                'issues': [],
                'warnings': ['此步骤输入为前一步实际结果，执行时按该结果校验'],
            })
    return {'steps': reports, 'atomic_commit': True, 'maximum_steps': 6, 'can_start_first': reports[0]['can_start']}
