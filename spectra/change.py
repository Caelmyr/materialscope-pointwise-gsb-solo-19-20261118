"""两时点重复采集变化分析。

设计原则：
- 不先各自平均。每个时点保留全部重复观测，逐像素估计组内均值与协方差，
  跨波段协方差（波段相关）来自整幅图像的混合组内估计并做收缩正则。
- 缺测不是零：逐波段按有效重复计数；有效重复不足或可比波段不足的像素
  明确标记为未决，并说明缺少什么证据。
- 不做亮度对齐：不估计也不扣除全局增益/偏移，整片亮度位移作为真实均值
  变化保留在检验中（Hotelling T^2 对均值位移友好）。
- 多重性按实际进入检验的像素集合做 Benjamini-Hochberg FDR 校正，不向
  未决像素外推同一可靠性。
"""
import copy

import numpy as np
from scipy import stats

from . import contracts as c
from .model import same_spectral_grid, same_spatial_grid, create_cube


POLICIES = {
    # 正式报告：尽量不出现假变化
    'formal': {
        'label': '正式报告（抑制假变化）',
        'fdr_level': 0.05,
        'minimum_repeats_per_group': 2,
        'minimum_common_bands': 3,
        'allow_diagonal_fallback': False,
        'minimum_covariance_support': 30,
        'minimum_degrees_of_freedom': 4,
        'minimum_effect_fraction_of_spread': 1.0,
        'require_effect_threshold': True,
    },
    # 复检筛查：接受少量误报以多找候选
    'screening': {
        'label': '复检筛查（多找候选，允许少量误报）',
        'fdr_level': 0.20,
        'minimum_repeats_per_group': 2,
        'minimum_common_bands': 2,
        'allow_diagonal_fallback': True,
        'minimum_covariance_support': 12,
        'minimum_degrees_of_freedom': 2,
        'minimum_effect_fraction_of_spread': 0.5,
        'require_effect_threshold': False,
    },
}

# 未决原因码 -> 说明与所需证据
UNDECIDED_REASONS = {
    'no_overlap': '两组没有任何共同有效波段：缺少可比较的同一波长观测',
    'insufficient_bands': '共同有效波段数低于该取向要求：需要更多波段同时在两组都有有效重复',
    'insufficient_repeats': '某组在该像素的有效重复次数不足：需要该时点更多有效重复采集',
    'insufficient_dof': '重复自由度不足，无法稳定估计差异标准误：需要每组更多重复',
    'covariance_unsupported': '图像内可用于估计波段协方差的完整光谱不足：需要更多无变化有效像素或更多重复',
    'degenerate_test': '收缩后的协方差仍奇异，多元检验不可用：需要更多重复或更多有效波段',
    'weak_evidence_below_effect': '统计上可分辨但幅度低于最小有意义变化，且现有波动不足以宣称稳定无变化：需要更多重复降低不确定性',
    'not_testable': '该像素没有任何有效观测，未进入检验',
}

DIRECTION_UP = 1
DIRECTION_DOWN = -1
DIRECTION_MIXED = 0
LABEL_NO_CHANGE = 0
LABEL_CHANGE_UP = 1
LABEL_CHANGE_DOWN = 2
LABEL_CHANGE_MIXED = 3
LABEL_UNDECIDED = 4


def validate_groups(before_cubes, after_cubes):
    if not 2 <= len(before_cubes) <= 12:
        raise c.Invalid('处理前重复组需要2–12幅影像')
    if not 2 <= len(after_cubes) <= 12:
        raise c.Invalid('处理后重复组需要2–12幅影像')
    reference = before_cubes[0]
    for cube in before_cubes[1:] + after_cubes:
        same_spectral_grid(reference, cube)
        same_spatial_grid(reference, cube)
        if cube.domain != reference.domain or cube.unit != reference.unit:
            raise c.Invalid('变化分析要求全部影像数值域与单位一致')
    total = sum(cube.values.nbytes + cube.valid.nbytes for cube in before_cubes + after_cubes)
    if total > 180000000:
        raise c.Invalid('变化分析工作集超过180MB，请先裁切')
    return reference


def _stack(cubes, selected):
    values = np.stack([cube.values[:, :, selected] for cube in cubes], axis=0).astype(np.float64)
    valid = np.stack([cube.valid[:, :, selected] for cube in cubes], axis=0)
    return values, valid


def _moments(values, valid, min_count):
    """逐像素逐波段的重复均值、无偏方差与有效重复数。缺测保持缺测。"""
    count = valid.sum(axis=0).astype(np.float64)
    summed = np.where(valid, values, 0.0).sum(axis=0)
    mean = np.divide(summed, count, out=np.zeros_like(summed), where=count > 0)
    deviation = np.where(valid, values - np.where(count > 0, mean, 0.0), 0.0)
    squared = (deviation ** 2).sum(axis=0)
    variance = np.divide(squared, np.maximum(count - 1, 1), out=np.zeros_like(squared), where=count > 1)
    usable = count >= min_count
    return mean, variance, count, usable


def _repeat_noise_covariance(before_v, before_valid, after_v, after_valid, var0, var1,
                             count0, count1, bands):
    """仅用组内重复残差估计重复噪声（不含材料空间差异）。

    返回：
    - noise_variance：逐波段重复噪声方差（在像素与重复上混合的无偏估计），
      这才是重复采集波动，材料在像素间的真实光谱差异不混入。
    - correlation：由组内离均差交叉项估计的波段相关矩阵（收缩后）。
    - support：参与估计的（像素, 重复, 波段）有效样本规模度量。
    """
    cross = np.zeros((bands, bands), dtype=np.float64)
    ss = np.zeros(bands, dtype=np.float64)
    weight_total = np.zeros(bands, dtype=np.float64)
    support = 0
    for values, valid, variance, count in (
        (before_v, before_valid, var0, count0),
        (after_v, after_valid, var1, count1),
    ):
        repeats, height, width, _ = values.shape
        flat = values.reshape(repeats, height * width, bands)
        flat_valid = valid.reshape(repeats, height * width, bands)
        # 逐波段有效重复次数（像素 x 波段）
        per_band_count = flat_valid.sum(axis=0)
        mean = np.divide(flat.sum(axis=0), np.maximum(per_band_count, 1),
                         out=np.zeros_like(flat[0]), where=per_band_count > 0)
        centered = np.where(flat_valid, flat - np.where(per_band_count > 0, mean, 0.0), 0.0)
        # 逐波段噪声方差：按各像素自由度(count-1)加权混合无偏样本方差
        for band in range(bands):
            dof = np.maximum(count[:, :, band].reshape(-1) - 1.0, 0.0)
            ss[band] += float((variance[:, :, band].reshape(-1) * dof).sum())
            weight_total[band] += float(dof.sum())
        # 波段对相关：使用两个波段在该像素都有 >=2 次有效重复的像素
        usable_pair = per_band_count >= 2  # (pixels, bands)
        for i in range(bands):
            for j in range(i, bands):
                pair_pixels = usable_pair[:, i] & usable_pair[:, j]
                if not pair_pixels.any():
                    continue
                ci = centered[:, pair_pixels, i]
                cj = centered[:, pair_pixels, j]
                acc = float((ci * cj).sum())
                cross[i, j] += acc
                cross[j, i] = cross[i, j]
        support += int(usable_pair.any(axis=1).sum())
    noise_variance = np.divide(ss, np.maximum(weight_total, 1.0))
    positive = noise_variance[noise_variance > 0]
    floor = max(float(np.median(positive)) * 1e-6 if positive.size else 1e-12, 1e-12)
    noise_variance = np.maximum(noise_variance, floor)
    # 相关矩阵：组内交叉项 / 组内平方项，避免材料空间差异污染
    diag_cross = np.maximum(np.diag(cross), floor)
    denom = np.sqrt(np.outer(diag_cross, diag_cross))
    correlation = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 0)
    correlation = np.clip(correlation, -0.999999, 0.999999)
    np.fill_diagonal(correlation, 1.0)
    correlation = (correlation + correlation.T) / 2.0
    # 对相关矩阵做收缩，小样本下向单位阵（独立波段）靠拢
    shrinkage = float(np.clip(1.0 - (support - bands - 1) / max(50.0, 4.0 * (bands + 1)), 0.05, 0.98))
    correlation = (1.0 - shrinkage) * correlation + shrinkage * np.eye(bands)
    covariance = correlation * np.sqrt(np.outer(noise_variance, noise_variance))
    return covariance, correlation, noise_variance, shrinkage, int(support), floor


def _band_selection(reference, options):
    if options.get('bands') is None:
        return list(range(reference.count))
    return c.index_list(options.get('bands'), reference.count, '分析波段', 1)


def _resolve_policy(policy_name, options):
    options = c.options_copy(options or {})
    policy_name = c.choice(policy_name, '误报取向', list(POLICIES))
    policy = copy.deepcopy(POLICIES[policy_name])
    if options.get('fdr_level') is not None:
        policy['fdr_level'] = c.number(options.get('fdr_level'), 'FDR水平', 0.001, 0.5)
    if options.get('minimum_effect_fraction_of_spread') is not None:
        policy['minimum_effect_fraction_of_spread'] = c.number(
            options.get('minimum_effect_fraction_of_spread'), '最小效应/波动比', 0.0, 10.0)
    return policy_name, policy, options


def compute_change_evidence(before_cubes, after_cubes, options=None):
    """计算与误报取向无关的变化证据（昂贵步骤，只做一次）。

    保留全部重复观测的逐像素均值、组内方差、有效计数、纯重复噪声协方差/相关，
    以及标准化差异、自由度和检验统计量。两种取向复用同一份证据，确保可比。
    """
    policy_name, _policy, options = _resolve_policy('formal', options)
    reference = validate_groups(before_cubes, after_cubes)
    selected = _band_selection(reference, options)
    bands_meta = [reference.bands[i] for i in selected]
    band_names = [item['name'] for item in bands_meta]
    waves = [item['wavelength_nm'] for item in bands_meta]
    n0, n1 = len(before_cubes), len(after_cubes)
    # 证据层用最宽松的最低重复数，资格门槛交给各取向
    min_repeats = min(POLICIES['formal']['minimum_repeats_per_group'],
                      POLICIES['screening']['minimum_repeats_per_group'])

    before_v, before_valid = _stack(before_cubes, selected)
    after_v, after_valid = _stack(after_cubes, selected)
    height, width, band_count = before_v.shape[1:]

    mean0, var0, count0, usable0 = _moments(before_v, before_valid, min_repeats)
    mean1, var1, count1, usable1 = _moments(after_v, after_valid, min_repeats)
    covariance, correlation, noise_variance, shrinkage, cov_support, floor = _repeat_noise_covariance(
        before_v, before_valid, after_v, after_valid, var0, var1, count0, count1, band_count)
    diag_spread = np.sqrt(np.maximum(noise_variance, floor))

    delta = mean1 - mean0
    with np.errstate(invalid='ignore', divide='ignore'):
        se0 = np.divide(var0, count0, out=np.zeros_like(var0), where=count0 > 0)
        se1 = np.divide(var1, count1, out=np.zeros_like(var1), where=count1 > 0)
        se = np.sqrt(np.maximum(se0 + se1, 0.0))

    common = usable0 & usable1
    common_band_count = common.sum(axis=2)
    safe_se = np.maximum(se, np.sqrt(floor))
    z = np.divide(delta, safe_se, out=np.zeros_like(delta), where=common)
    z_sq = z ** 2

    min_c0 = np.where(common.any(axis=2), np.where(common, count0, np.inf).min(axis=2), 0.0)
    min_c1 = np.where(common.any(axis=2), np.where(common, count1, np.inf).min(axis=2), 0.0)
    dof_map = np.maximum((min_c0 - 1) + (min_c1 - 1), 0.0)

    full_invertible = True
    try:
        cov_inv_full = np.linalg.inv(covariance)
        if not np.all(np.isfinite(cov_inv_full)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        full_invertible = False
        cov_inv_full = np.diag(np.divide(1.0, np.maximum(np.diag(covariance), floor)))

    # 两样本 Hotelling T^2：
    #   T^2 = (n0*n1/(n0+n1)) * d' S^{-1} d，d 为原始尺度均值差，S 为组内协方差。
    # 逐像素有效重复数可能不同，系数按该像素可比波段上的最小重复数取。
    t2 = np.zeros((height, width), dtype=np.float64)
    method_map = np.full((height, width), '', dtype=object)
    stat_valid = np.zeros((height, width), dtype=bool)
    eff_n0 = np.where(common.any(axis=2),
                      np.where(common, count0, np.inf).min(axis=2), 0.0)
    eff_n1 = np.where(common.any(axis=2),
                      np.where(common, count1, np.inf).min(axis=2), 0.0)
    harmonic = np.divide(eff_n0 * eff_n1, np.maximum(eff_n0 + eff_n1, 1e-12),
                         out=np.zeros((height, width)), where=(eff_n0 > 0) & (eff_n1 > 0))
    full_mask = common.all(axis=2)
    if full_mask.any() and full_invertible:
        df = delta[full_mask]
        quad = np.einsum('ni,ij,nj->n', df, cov_inv_full, df)
        t2[full_mask] = harmonic[full_mask] * quad
        method_map[full_mask] = 'hotelling'
        stat_valid[full_mask] = True
    elif full_mask.any():
        quad = np.einsum('ni,ij,nj->n', delta[full_mask], cov_inv_full, delta[full_mask])
        t2[full_mask] = harmonic[full_mask] * quad
        method_map[full_mask] = 'diagonal'
        stat_valid[full_mask] = True

    partial_mask = (~common.all(axis=2)) & (common_band_count > 0)
    delta_flat = delta.reshape(-1, band_count)
    t2_flat = t2.reshape(-1)
    partial_flat = partial_mask.reshape(-1)
    if partial_mask.any():
        keys = common.reshape(-1, band_count)[partial_flat].astype(np.uint8)
        bit_vectors = (np.uint64(keys) * np.uint64(1) << np.arange(band_count, dtype=np.uint64))
        bits = bit_vectors.sum(axis=1, dtype=np.uint64)
        unique_bits = np.unique(bits)
        partial_positions = np.flatnonzero(partial_flat)
        for bit in unique_bits:
            member_pos = partial_positions[bits == bit]
            band_flags = np.array([(int(bit) >> b) & 1 for b in range(band_count)], dtype=bool)
            bands_idx = np.flatnonzero(band_flags)
            sub_cov = covariance[np.ix_(bands_idx, bands_idx)]
            use_diagonal = False
            try:
                sub_inv = np.linalg.inv(sub_cov)
                if not np.all(np.isfinite(sub_inv)):
                    raise np.linalg.LinAlgError
            except np.linalg.LinAlgError:
                use_diagonal = True
                sub_inv = np.diag(np.divide(1.0, np.maximum(np.diag(sub_cov), floor)))
            dm = delta_flat[member_pos][:, bands_idx]
            quad = np.einsum('ni,ij,nj->n', dm, sub_inv, dm)
            method = 'diagonal' if use_diagonal else 'hotelling'
            t2_flat[member_pos] = harmonic.reshape(-1)[member_pos] * quad
            method_map.reshape(-1)[member_pos] = method
            stat_valid.reshape(-1)[member_pos] = True

    # 两样本 Hotelling T^2 -> F：
    #   F = T^2 * (nu - p + 1) / (p * nu)，df1=p，df2=nu-p+1，nu=n0+n1-2
    pvalue = np.full((height, width), np.nan, dtype=np.float64)
    if stat_valid.any():
        p_dim = np.where(stat_valid, common_band_count, 1).astype(np.float64)
        nu = eff_n0 + eff_n1 - 2.0
        df2 = nu - p_dim + 1.0
        usable_f = stat_valid & (df2 > 0) & (nu > 0)
        f_stat = np.divide(t2 * df2, p_dim * np.maximum(nu, 1e-12),
                           out=np.zeros_like(t2), where=usable_f)
        pvalue = np.where(usable_f, stats.f.sf(np.maximum(f_stat, 0.0), p_dim, df2), np.nan)
        # df2 不可用时退回卡方（极少出现，仅大波段极小样本）
        chi2_fallback = stat_valid & ~usable_f
        if chi2_fallback.any():
            pvalue = np.where(chi2_fallback,
                              stats.chi2.sf(np.maximum(t2, 0.0), p_dim), pvalue)
            method_map[chi2_fallback] = 'diagonal'
    pvalue = np.clip(pvalue, 1e-300, 1.0)

    with np.errstate(invalid='ignore'):
        rms_effect = np.sqrt(np.where(common_band_count > 0,
                                      np.where(common, delta ** 2, 0.0).sum(axis=2) /
                                      np.maximum(common_band_count, 1), 0.0))
        std_effect = np.sqrt(np.where(common_band_count > 0,
                                      np.where(common, z_sq, 0.0).sum(axis=2) /
                                      np.maximum(common_band_count, 1), 0.0))
    pos_count = np.where(common, z > 0, False).sum(axis=2)
    neg_count = np.where(common, z < 0, False).sum(axis=2)
    weights = np.divide(1.0, safe_se ** 2, out=np.zeros_like(safe_se), where=common)
    signed = np.divide(np.where(common, z * weights, 0.0).sum(axis=2),
                       np.where(common, weights, 0.0).sum(axis=2),
                       out=np.zeros((height, width)), where=common_band_count > 0)
    direction = np.where(pos_count & neg_count, DIRECTION_MIXED,
                         np.where(signed >= 0, DIRECTION_UP, DIRECTION_DOWN)).astype(np.int8)

    return {
        'reference': reference,
        'before_cubes': before_cubes,
        'after_cubes': after_cubes,
        'selected': selected,
        'band_names': band_names,
        'waves': waves,
        'n0': n0,
        'n1': n1,
        'height': height,
        'width': width,
        'band_count': band_count,
        'delta': delta,
        'se': se,
        'z': z,
        'z_sq': z_sq,
        'common': common,
        'common_band_count': common_band_count,
        'count0': count0,
        'count1': count1,
        'eff_n0': eff_n0,
        'eff_n1': eff_n1,
        'dof_map': dof_map,
        't2': t2,
        'pvalue': pvalue,
        'method_map': method_map,
        'stat_valid': stat_valid,
        'rms_effect': rms_effect,
        'std_effect': std_effect,
        'direction': direction,
        'covariance': covariance,
        'correlation': correlation,
        'noise_variance': noise_variance,
        'shrinkage': shrinkage,
        'cov_support': cov_support,
        'floor': floor,
        'diag_spread': diag_spread,
        'full_invertible': full_invertible,
    }


def classify_change(evidence, policy_name, options=None):
    """在同一份证据上按指定误报取向分类，返回 (立方体, 报告, 方案)。"""
    policy_name, policy, options = _resolve_policy(policy_name, options)
    reference = evidence['reference']
    height, width = evidence['height'], evidence['width']
    common = evidence['common']
    common_band_count = evidence['common_band_count']
    dof_map = evidence['dof_map']
    count0, count1 = evidence['count0'], evidence['count1']
    eff_n0, eff_n1 = evidence['eff_n0'], evidence['eff_n1']
    t2, pvalue = evidence['t2'], evidence['pvalue']
    rms_effect, std_effect = evidence['rms_effect'], evidence['std_effect']
    direction = evidence['direction']
    method_map = evidence['method_map']
    stat_valid = evidence['stat_valid']
    cov_support, shrinkage = evidence['cov_support'], evidence['shrinkage']
    floor, diag_spread = evidence['floor'], evidence['diag_spread']
    delta, z_sq = evidence['delta'], evidence['z_sq']
    correlation = evidence['correlation']

    # Hotelling T^2 可识别条件：该像素可比波段数 p 须满足 nu=n0+n1-2 >= p
    nu_map = eff_n0 + eff_n1 - 2.0
    identifiable = nu_map >= common_band_count

    # ---- 检验资格与未决原因（带优先级） ----
    reason = np.full((height, width), 'not_testable', dtype=object)
    reason[(common_band_count == 0) &
           ((count0.sum(axis=2) == 0) | (count1.sum(axis=2) == 0))] = 'no_overlap'
    reason[(common_band_count == 0) & (reason == 'not_testable')] = 'insufficient_repeats'
    reason[(common_band_count > 0) &
           (common_band_count < policy['minimum_common_bands'])] = 'insufficient_bands'
    meets_bands = common_band_count >= policy['minimum_common_bands']
    reason[meets_bands & (dof_map < policy['minimum_degrees_of_freedom'])] = 'insufficient_dof'
    # 自由度虽达取向门槛，但相对可比波段数仍不足以识别多元检验
    reason[meets_bands & (dof_map >= policy['minimum_degrees_of_freedom']) &
           ~identifiable] = 'insufficient_dof'
    meets_dof = meets_bands & (dof_map >= policy['minimum_degrees_of_freedom']) & identifiable
    if cov_support < policy['minimum_covariance_support']:
        reason[meets_dof] = 'covariance_unsupported'
        meets_dof &= False
    if not evidence['full_invertible'] and not policy['allow_diagonal_fallback']:
        reason[meets_dof] = 'degenerate_test'
        meets_dof &= False
    eligible = meets_dof & stat_valid
    # 统计量不可用（退化且该取向不允许对角回退）
    reason[meets_dof & ~stat_valid] = 'degenerate_test'

    tested = eligible
    tested_idx = np.flatnonzero(tested)

    # Benjamini-Hochberg FDR，仅对实际进入检验的像素
    reject_map = np.zeros((height, width), dtype=bool)
    bh_threshold = None
    achievable = True
    if len(tested_idx):
        pvals = pvalue.ravel()[tested_idx]
        order = np.argsort(pvals, kind='stable')
        ranked = pvals[order]
        m = len(pvals)
        critical = policy['fdr_level'] * (np.arange(1, m + 1) / m)
        passed = ranked <= critical
        if passed.any():
            cutoff_rank = int(np.flatnonzero(passed).max())
            rejected_local = np.zeros(m, dtype=bool)
            rejected_local[order[:cutoff_rank + 1]] = True
            reject_map.ravel()[tested_idx] = rejected_local
            bh_threshold = float(ranked[order[cutoff_rank]])
        else:
            bh_threshold = float(critical[0])
            achievable = False

    # 最小有意义效应阈值
    effect_ratio = policy['minimum_effect_fraction_of_spread']
    band_threshold = diag_spread[None, None, :] * effect_ratio
    over_threshold_per_band = common & (np.abs(delta) >= band_threshold)
    fraction_over = np.divide(over_threshold_per_band.sum(axis=2),
                              np.maximum(common_band_count, 1),
                              out=np.zeros((height, width)), where=common_band_count > 0)
    meets_effect = (std_effect >= effect_ratio) & (fraction_over >= 0.5)

    label = np.full((height, width), LABEL_UNDECIDED, dtype=np.int8)
    label[tested & ~reject_map] = LABEL_NO_CHANGE
    if policy['require_effect_threshold']:
        significant_but_small = tested & reject_map & ~meets_effect
        label[significant_but_small] = LABEL_UNDECIDED
        reason[significant_but_small] = 'weak_evidence_below_effect'
        confirmed = tested & reject_map & meets_effect
    else:
        confirmed = tested & reject_map
    label[confirmed & (direction == DIRECTION_UP)] = LABEL_CHANGE_UP
    label[confirmed & (direction == DIRECTION_DOWN)] = LABEL_CHANGE_DOWN
    label[confirmed & (direction == DIRECTION_MIXED)] = LABEL_CHANGE_MIXED

    undecided = label == LABEL_UNDECIDED
    no_change = label == LABEL_NO_CHANGE

    protocol = _build_protocol(
        evidence['before_cubes'], evidence['after_cubes'], evidence['selected'],
        evidence['band_names'], evidence['waves'], policy, policy_name,
        evidence['covariance'], correlation, evidence['noise_variance'], shrinkage,
        cov_support, floor, evidence['n0'], evidence['n1'], reference)
    report = _build_report(
        policy, policy_name, label, reason, tested, confirmed, undecided, no_change,
        reject_map, meets_effect, pvalue, t2, rms_effect, std_effect, common_band_count,
        count0, count1, dof_map, cov_support, shrinkage, bh_threshold, achievable,
        evidence['band_names'], evidence['waves'], evidence['n0'], evidence['n1'],
        reference, protocol, method_map)
    cubes = _build_cubes(reference, label, undecided, confirmed, t2, rms_effect, std_effect,
                         common, report, protocol)
    return cubes, report, protocol


def analyze_change(before_cubes, after_cubes, policy_name, options=None):
    """单次便捷入口：计算证据并按指定取向分类。"""
    evidence = compute_change_evidence(before_cubes, after_cubes, options)
    return classify_change(evidence, policy_name, options)


def _downsample_labels(label, maximum=72):
    """确定性最近邻降采样标签图，供前端小图预览（不用于导出或结论）。"""
    height, width = label.shape
    step = max(1, int(np.ceil(max(height, width) / maximum)))
    sampled = label[::step, ::step]
    return sampled.astype(int).tolist(), step


def preview_change_policies(before_cubes, after_cubes, options=None):
    """在同一份重复证据上对比两种取向，不落盘。返回紧凑对比与降采样标签。"""
    evidence = compute_change_evidence(before_cubes, after_cubes, options)
    comparison = {}
    for policy_name in ('formal', 'screening'):
        cubes, report, protocol = classify_change(evidence, policy_name, options)
        label = cubes[1].values[:, :, 0].astype(np.int8)
        undecided = cubes[3].values[:, :, 0].astype(np.int8)
        label_grid, step = _downsample_labels(label)
        undecided_grid, _ = _downsample_labels(undecided)
        comparison[policy_name] = {
            'policy_label': POLICIES[policy_name]['label'],
            'fdr_level': report['fdr_support']['declared_fdr_level'],
            'pixel_accounting': report['pixel_accounting'],
            'fdr_support': report['fdr_support'],
            'confirmed_magnitude': report['confirmed_magnitude'],
            'effect_threshold': report['effect_threshold'],
            'covariance_estimate': report['covariance_estimate'],
            'label_preview': label_grid,
            'undecided_preview': undecided_grid,
            'preview_step': step,
            'label_legend': cubes[1].metadata['label_legend'],
        }
    return {
        'schema': 'materialscope_change_preview_v1',
        'image_size': [evidence['height'], evidence['width']],
        'analysis_band_names': evidence['band_names'],
        'analysis_wavelengths_nm': evidence['waves'],
        'group_repeats': {'before': evidence['n0'], 'after': evidence['n1']},
        'shared_evidence': {
            'covariance_support_pixels': evidence['cov_support'],
            'shrinkage': evidence['shrinkage'],
            'estimator': evidence['reference'].domain,
            'brightness_alignment': 'none',
            'missing_policy': 'excluded per pixel, never zero',
        },
        'policies': comparison,
        'undecided_reason_dictionary': copy.deepcopy(UNDECIDED_REASONS),
        'note': '两种取向复用完全相同的重复观测证据；确认一种后才生成并固化结果。',
    }


def _build_protocol(before_cubes, after_cubes, selected, band_names, waves, policy,
                    policy_name, covariance, correlation, noise_variance, shrinkage,
                    cov_support, floor, n0, n1, reference):
    return {
        'schema': 'materialscope_change_protocol_v1',
        'policy': policy_name,
        'policy_label': policy['label'],
        'fdr_level': policy['fdr_level'],
        'minimum_repeats_per_group': policy['minimum_repeats_per_group'],
        'minimum_common_bands': policy['minimum_common_bands'],
        'minimum_covariance_support': policy['minimum_covariance_support'],
        'minimum_degrees_of_freedom': policy['minimum_degrees_of_freedom'],
        'minimum_effect_fraction_of_spread': policy['minimum_effect_fraction_of_spread'],
        'allow_diagonal_fallback': policy['allow_diagonal_fallback'],
        'require_effect_threshold': policy['require_effect_threshold'],
        'analysis_band_indices': list(selected),
        'analysis_band_names': band_names,
        'analysis_wavelengths_nm': waves,
        'before': {'repeat_count': n0, 'asset_fingerprints': [c.fingerprint() for c in before_cubes]},
        'after': {'repeat_count': n1, 'asset_fingerprints': [c.fingerprint() for c in after_cubes]},
        'covariance': {
            'shrinkage': shrinkage,
            'support_pixels': int(cov_support),
            'variance_floor': floor,
            'repeat_noise_variance': np.asarray(noise_variance).tolist(),
            'matrix': covariance.tolist(),
            'correlation': correlation.tolist(),
            'estimator': 'within-repeat residuals only; between-pixel material differences excluded',
        },
        'grid': copy.deepcopy(reference.grid),
        'domain': reference.domain,
        'unit': reference.unit,
        'brightness_alignment': 'none: global offset/gain is not removed, whole-scene shifts remain detectable changes',
        'missing_policy': 'missing bands are excluded per pixel; never treated as zero',
        'multiplicity': 'Benjamini-Hochberg FDR over the actual set of tested pixels only',
    }


def _stats_on(mask, field):
    vals = field[mask]
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return None
    return {'min': float(vals.min()), 'median': float(np.median(vals)),
            'mean': float(vals.mean()), 'max': float(vals.max())}


def _build_report(policy, policy_name, label, reason, tested, confirmed, undecided, no_change,
                  reject_map, meets_effect, pvalue, t2, rms_effect, std_effect, common_band_count,
                  count0, count1, dof_map, cov_support, shrinkage, bh_threshold, achievable,
                  band_names, waves, n0, n1, reference, protocol, method_map):
    reason_counts = {code: int(np.sum(undecided & (reason == code))) for code in UNDECIDED_REASONS}
    tested_count = int(tested.sum())
    stable = tested & ~confirmed
    fdr_support = {
        'declared_fdr_level': policy['fdr_level'],
        'tested_pixels': tested_count,
        'bh_pvalue_threshold': bh_threshold,
        'level_achievable': bool(achievable),
        'note': ('FDR 仅在实际进入检验的像素集合上控制；未决像素不参与、不享有该可靠性'
                 if achievable else
                 '现有重复波动下没有像素能通过该 FDR 水平，所声明误报水平缺乏证据支持，相关位置保持未决'),
    }
    return {
        'schema': 'materialscope_change_report_v1',
        'policy': policy_name,
        'policy_label': policy['label'],
        'image_size': [reference.height, reference.width],
        'band_count_analyzed': len(band_names),
        'bands': [{'name': n, 'wavelength_nm': w} for n, w in zip(band_names, waves)],
        'group_repeats': {'before': n0, 'after': n1},
        'pixel_accounting': {
            'total_pixels': int(label.size),
            'tested_pixels': tested_count,
            'confirmed_change_pixels': int(confirmed.sum()),
            'change_up_pixels': int(np.sum(label == LABEL_CHANGE_UP)),
            'change_down_pixels': int(np.sum(label == LABEL_CHANGE_DOWN)),
            'change_mixed_direction_pixels': int(np.sum(label == LABEL_CHANGE_MIXED)),
            'no_change_evidence_pixels': int(no_change.sum()),
            'undecided_pixels': int(undecided.sum()),
            'undecided_reason_counts': reason_counts,
        },
        'fdr_support': fdr_support,
        'stability_evidence': {
            'stable_evidence_pixels': int(stable.sum()),
            'std_effect_distribution': _stats_on(stable, std_effect),
            'rms_change_distribution': _stats_on(stable, rms_effect),
            'interpretation': '通过检验但未达确认标准的像素；标准化效应越小代表无变化证据越强',
        },
        'confirmed_magnitude': {
            'mahalanobis_t2': _stats_on(confirmed, t2),
            'rms_change_in_unit': _stats_on(confirmed, rms_effect),
            'standardized_effect': _stats_on(confirmed, std_effect),
        },
        'covariance_estimate': {
            'support_pixels': int(cov_support),
            'shrinkage': float(shrinkage),
            'band_correlation_used': True,
            'diagonal_fallback_pixels': int(np.sum(method_map == 'diagonal')),
        },
        'effect_threshold': {
            'fraction_of_repeat_spread': policy['minimum_effect_fraction_of_spread'],
            'enforced': bool(policy['require_effect_threshold']),
        },
        'undecided_reason_dictionary': copy.deepcopy(UNDECIDED_REASONS),
        'no_brightness_alignment': protocol['brightness_alignment'],
        'no_missing_as_zero': protocol['missing_policy'],
    }


def _build_cubes(reference, label, undecided, confirmed, t2, rms_effect, std_effect,
                 common, report, protocol):
    grid = reference.grid
    any_common = common.any(axis=2)
    mag_values = np.stack([t2, rms_effect, std_effect], axis=2).astype(np.float32)
    mag_valid = np.repeat(any_common[:, :, None], 3, axis=2)
    mag_bands = [
        {'name': 'Hotelling T2 statistic', 'wavelength_nm': None},
        {'name': 'RMS change [' + reference.unit + ']', 'wavelength_nm': None},
        {'name': 'Standardized effect size', 'wavelength_nm': None},
    ]
    magnitude = create_cube(
        mag_values, mag_valid, mag_bands, 'derived', reference.unit, grid,
        {'operation': 'change_magnitude', 'report': c.json_safe(report),
         'protocol': c.json_safe(protocol)})

    decision = create_cube(
        label.astype(np.float32)[:, :, None], np.ones((reference.height, reference.width, 1), dtype=bool),
        [{'name': 'Change decision label', 'wavelength_nm': None}],
        'labels', 'class', grid,
        {'operation': 'change_decision', 'label_legend': {
            '0': 'no change (evidence of stability)',
            '1': 'confirmed change, increase',
            '2': 'confirmed change, decrease',
            '3': 'confirmed change, mixed direction across bands',
            '4': 'undecided (insufficient evidence)',
        }, 'protocol': c.json_safe(protocol), 'report': c.json_safe(report)})

    undecided_cube = create_cube(
        undecided.astype(np.float32)[:, :, None],
        np.ones((reference.height, reference.width, 1), dtype=bool),
        [{'name': 'Undecided mask', 'wavelength_nm': None}],
        'labels', 'mask', grid,
        {'operation': 'change_undecided_mask', 'protocol': c.json_safe(protocol),
         'report': c.json_safe(report)})

    confirmed_cube = create_cube(
        confirmed.astype(np.float32)[:, :, None],
        np.ones((reference.height, reference.width, 1), dtype=bool),
        [{'name': 'Confirmed change mask', 'wavelength_nm': None}],
        'labels', 'mask', grid,
        {'operation': 'change_confirmed_mask', 'protocol': c.json_safe(protocol),
         'report': c.json_safe(report)})
    return [magnitude, decision, confirmed_cube, undecided_cube]


def pixel_evidence(before_cubes, after_cubes, protocol, x, y):
    """页面点选：回看两组实测光谱、波动与变化依据。"""
    reference = validate_groups(before_cubes, after_cubes)
    x = c.integer(x, '横坐标', 0, reference.width - 1)
    y = c.integer(y, '纵坐标', 0, reference.height - 1)
    selected = protocol['analysis_band_indices']

    def group_rows(cubes):
        rows = []
        for index, cube in enumerate(cubes):
            vals, valids = [], []
            for b in selected:
                ok = bool(cube.valid[y, x, b])
                valids.append(ok)
                vals.append(float(cube.values[y, x, b]) if ok else None)
            rows.append({'measurement_index': index, 'fingerprint': cube.fingerprint(),
                         'values': vals, 'valid': valids, 'valid_band_count': int(sum(valids))})
        return rows

    before_rows = group_rows(before_cubes)
    after_rows = group_rows(after_cubes)

    def aggregate(rows):
        means, sds, counts = [], [], []
        for bi in range(len(selected)):
            vals = [r['values'][bi] for r in rows if r['valid'][bi] and r['values'][bi] is not None]
            counts.append(len(vals))
            if vals:
                means.append(float(np.mean(vals)))
                sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            else:
                means.append(None)
                sds.append(None)
        return means, sds, counts

    b_mean, b_sd, b_count = aggregate(before_rows)
    a_mean, a_sd, a_count = aggregate(after_rows)
    delta, per_band_z = [], []
    for bi in range(len(selected)):
        if b_mean[bi] is None or a_mean[bi] is None:
            delta.append(None)
            per_band_z.append(None)
            continue
        d = a_mean[bi] - b_mean[bi]
        delta.append(d)
        se2 = (b_sd[bi] ** 2 / b_count[bi] if b_count[bi] else 0.0) + \
              (a_sd[bi] ** 2 / a_count[bi] if a_count[bi] else 0.0)
        per_band_z.append(float(d / np.sqrt(se2)) if se2 > 0 else None)

    common_bands = [bi for bi in range(len(selected))
                    if b_count[bi] >= protocol['minimum_repeats_per_group']
                    and a_count[bi] >= protocol['minimum_repeats_per_group']]
    return {
        'schema': 'materialscope_change_evidence_v1',
        'x': x, 'y': y, 'world': reference.pixel_center(x, y),
        'analysis_band_names': protocol['analysis_band_names'],
        'analysis_wavelengths_nm': protocol['analysis_wavelengths_nm'],
        'unit': reference.unit, 'domain': reference.domain,
        'before_repeats': before_rows, 'after_repeats': after_rows,
        'before_mean': b_mean, 'before_sample_sd': b_sd, 'before_valid_counts': b_count,
        'after_mean': a_mean, 'after_sample_sd': a_sd, 'after_valid_counts': a_count,
        'delta_after_minus_before': delta,
        'per_band_standardized_difference': per_band_z,
        'common_comparable_band_count': len(common_bands),
        'policy': protocol['policy'], 'policy_label': protocol['policy_label'],
        'fdr_level': protocol['fdr_level'],
        'brightness_alignment': protocol['brightness_alignment'],
        'missing_policy': protocol['missing_policy'],
    }
