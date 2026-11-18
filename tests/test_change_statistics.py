"""变化分析统计健全性测试（合成数据，独立于 Web 层）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from spectra.model import create_cube
from spectra import change


def make_group(base, repeats, noise, rng, valid=None, offset=0.0):
    cubes = []
    h, w, b = base.shape
    for _ in range(repeats):
        values = base + offset + rng.normal(0, noise, base.shape)
        v = np.ones(base.shape, dtype=bool) if valid is None else valid.copy()
        cubes.append(create_cube(values.astype(np.float32), v,
                                 [{'name': f'B{i}', 'wavelength_nm': 400 + 50 * i} for i in range(b)],
                                 'reflectance', 'fraction',
                                 {'x_step': 0.1, 'y_step': 0.1, 'origin': [0, 0], 'unit': 'mm'}, {}))
    return cubes


def main():
    rng = np.random.default_rng(1)
    h, w, b = 40, 40, 6
    # 带波段相关的基础光谱场
    latent = rng.normal(0, 1, (h, w, 1))
    base = 0.5 + 0.05 * latent + np.linspace(-0.05, 0.05, b)[None, None, :]
    noise = 0.02

    # 1) 无变化零假设：两组来自同分布（6 次重复，F 自由度充分）
    before = make_group(base, 6, noise, rng)
    after_null = make_group(base, 6, noise, rng)
    cubes, report, protocol = change.analyze_change(before, after_null, 'formal', {})
    label = cubes[1].values[:, :, 0]
    false_rate = (label == change.LABEL_CHANGE_UP).sum() + (label == change.LABEL_CHANGE_DOWN).sum()
    fp_fraction = false_rate / (h * w)
    print('[null] formal confirmed false changes:', int(false_rate), '/', h * w,
          f'({fp_fraction:.3f})', 'undecided:', report['pixel_accounting']['undecided_pixels'])
    # 正式取向 FDR 目标 5%，独立零假设下应近乎零误报
    assert fp_fraction <= 0.05, 'Formal policy should control false positives near 5% FDR'
    assert report['fdr_support']['declared_fdr_level'] == 0.05

    cubes_s, report_s, _ = change.analyze_change(before, after_null, 'screening', {})
    label_s = cubes_s[1].values[:, :, 0]
    fp_s = np.isin(label_s, [1, 2, 3]).sum()
    fp_s_fraction = fp_s / (h * w)
    print('[null] screening confirmed:', int(fp_s), f'({fp_s_fraction:.3f}, target FDR 20%)',
          'undecided:', report_s['pixel_accounting']['undecided_pixels'])
    # 筛查取向候选不少于正式取向，且独立零假设下不应远超其 FDR 目标
    assert fp_s >= false_rate
    assert fp_s_fraction <= 0.30

    # 2) 真实局部变化：右半区域 +0.15（远超噪声），6 次重复保证检出力
    after_change = make_group(base, 6, noise, rng)
    changed_region = np.zeros((h, w), dtype=bool)
    changed_region[:, w // 2:] = True
    new_cubes = []
    for cube in after_change:
        vals = cube.values.copy()
        vals[changed_region] += 0.15
        new_cubes.append(create_cube(vals, cube.valid, list(cube.bands), cube.domain, cube.unit, cube.grid, {}))
    cubes2, report2, protocol2 = change.analyze_change(before, new_cubes, 'formal', {})
    label2 = cubes2[1].values[:, :, 0]
    detected = np.isin(label2, [1, 2, 3])
    power = detected[changed_region].mean()
    false_left = detected[~changed_region].mean()
    print('[change] formal power on changed region:', round(float(power), 3),
          'false rate on unchanged region:', round(float(false_left), 4))
    assert power > 0.8, 'Should detect strong real change with adequate repeats'
    assert false_left < 0.08
    # 方向应为上升
    assert (label2[changed_region] == change.LABEL_CHANGE_UP).mean() > 0.8

    # 3) 整片亮度位移（全局 offset）不应被消掉
    after_offset = make_group(base, 6, noise, rng, offset=0.12)
    cubes3, report3, _ = change.analyze_change(before, after_offset, 'formal', {})
    label3 = cubes3[1].values[:, :, 0]
    global_detected = np.isin(label3, [1, 2, 3]).mean()
    print('[global offset] detected fraction:', round(float(global_detected), 3))
    assert global_detected > 0.5, 'Whole-scene brightness shift must remain detectable'

    # 4) 缺测：一组某波段在一块区域全部缺测，不能当零
    valid = np.ones(base.shape, dtype=bool)
    valid[5:12, 5:12, 2] = False
    before_missing = make_group(base, 6, noise, rng, valid=valid)
    cubes4, report4, protocol4 = change.analyze_change(before_missing, after_null, 'formal', {})
    label4 = cubes4[1].values[:, :, 0]
    # 缺测像素不应被误判为下降变化
    missing_block = ~valid[:, :, 2]
    bad = (label4[missing_block] == change.LABEL_CHANGE_DOWN).sum()
    print('[missing] false down-change in missing band block:', int(bad),
          'undecided reasons:', report4['pixel_accounting']['undecided_reason_counts'])
    assert bad == 0

    # 5) 重复不足：每组 2 次（nu=2 < 波段数 6），Hotelling 不可识别 -> 全部未决
    before_2 = make_group(base, 2, noise, rng)
    after_2 = make_group(base, 2, noise, rng)
    cubes5, report5, _ = change.analyze_change(before_2, after_2, 'formal', {})
    print('[low dof] formal undecided:', report5['pixel_accounting']['undecided_pixels'],
          report5['pixel_accounting']['undecided_reason_counts'])
    assert report5['pixel_accounting']['undecided_pixels'] == h * w
    assert report5['pixel_accounting']['undecided_reason_counts']['insufficient_dof'] == h * w
    cubes5b, report5b, _ = change.analyze_change(before_2, after_2, 'screening', {})
    print('[low dof] screening tested:', report5b['pixel_accounting']['tested_pixels'],
          'undecided:', report5b['pixel_accounting']['undecided_pixels'])
    # 筛查取向也不能绕过 Hotelling 可识别性
    assert report5b['pixel_accounting']['tested_pixels'] == 0

    # 5b) 只剩无法比较的波段：formal 需 >=3 共同波段，构造只剩 1-2 个
    valid_few = np.ones(base.shape, dtype=bool)
    valid_few[:, :, 2:] = False  # 只保留前 2 个波段有效
    before_few = make_group(base, 4, noise, rng, valid=valid_few)
    after_few = make_group(base, 4, noise, rng, valid=valid_few)
    cubes5c, report5c, _ = change.analyze_change(before_few, after_few, 'formal', {})
    few_reasons = report5c['pixel_accounting']['undecided_reason_counts']
    print('[few bands] formal undecided reasons:', few_reasons)
    assert few_reasons['insufficient_bands'] == h * w
    # screening 只需 2 个共同波段，且 nu=6 >= p=2 可识别，应可检验
    cubes5d, report5d, _ = change.analyze_change(before_few, after_few, 'screening', {})
    print('[few bands] screening tested:', report5d['pixel_accounting']['tested_pixels'])
    assert report5d['pixel_accounting']['tested_pixels'] == h * w

    # 5c) 完全无重叠：一组某像素全部缺测
    valid_none = np.ones(base.shape, dtype=bool)
    valid_none[20:25, 20:25, :] = False
    before_none = make_group(base, 6, noise, rng, valid=valid_none)
    cubes5e, report5e, _ = change.analyze_change(before_none, after_null, 'formal', {})
    none_reasons = report5e['pixel_accounting']['undecided_reason_counts']
    print('[no overlap] undecided reasons:', none_reasons)
    assert none_reasons['no_overlap'] == 25

    # 6) 点选证据回看
    evidence = change.pixel_evidence(before, new_cubes, protocol2, w // 2 + 3, h // 2)
    assert len(evidence['before_repeats']) == 6
    assert evidence['common_comparable_band_count'] == 6
    assert all(v is not None for v in evidence['delta_after_minus_before'])
    print('[evidence] sample delta at changed pixel:',
          [round(v, 3) for v in evidence['delta_after_minus_before']])

    # 7) 协议固化：指纹与波段固定
    assert protocol2['before']['repeat_count'] == 6
    assert len(protocol2['before']['asset_fingerprints']) == 6
    assert protocol2['analysis_band_indices'] == list(range(b))
    print('[protocol] frozen fingerprints and bands OK')

    print('\nALL CHANGE-ANALYSIS STATISTICAL TESTS PASSED')


if __name__ == '__main__':
    main()
