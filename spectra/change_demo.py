"""两时点变化分析的程序合成示范。

生成处理前/后两组各若干次同网格、同波长、同单位的重复反射率采集，
刻意包含：真实局部材料变化、整片亮度位移、逐波段缺测块、以及一组重复
不足的区域，用于展示检出、方向、幅度与未决证据。全部为程序合成，不是
现实观测。
"""
import numpy as np

from .demo import material_signatures
from .model import create_cube
from . import contracts as c


def create_change_demo(width=160, height=120, bands=6, repeats=6, seed=91):
    width = c.integer(width, '示范宽度', 64, 512)
    height = c.integer(height, '示范高度', 64, 512)
    bands = c.integer(bands, '示范波段数', 4, 24)
    repeats = c.integer(repeats, '每组重复数', 2, 8)
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(450, 950, bands)
    signatures = material_signatures(wavelengths)
    yy, xx = np.indices((height, width), dtype=float)
    nx, ny = xx / width, yy / height

    labels = np.zeros((height, width), dtype=int)
    labels[(nx > 0.32) & (nx < 0.66)] = 1
    labels[nx >= 0.66] = 2
    base = signatures[labels]
    base += (0.012 * np.sin(xx / 2.4) * np.cos(yy / 3.7))[:, :, None]

    band_rows = [{'name': f'{wave:.0f} nm', 'wavelength_nm': float(wave), 'fwhm_nm': 22.0}
                 for wave in wavelengths]
    grid = {'x_step': 0.08, 'y_step': 0.08, 'origin': [0, 0], 'unit': 'mm'}
    meta = {'synthetic': True, 'generator': 'materialscope change demo v1',
            'seed': seed, 'not_real_observation': True}
    noise = 0.01

    # 真实局部变化：椭圆形异物区，处理后在长波方向抬升（材料变化，非亮度）
    change_mask = (((nx - 0.30) ** 2 / 0.06 ** 2 + (ny - 0.68) ** 2 / 0.08 ** 2) < 1)
    spectral_change = np.zeros(bands)
    spectral_change[bands // 2:] = 0.14  # 仅长波波段上升 -> 方向可辨的材料变化
    after_true = base + np.where(change_mask[:, :, None], spectral_change[None, None, :], 0.0)

    # 整片亮度位移：处理后整体轻微变暗（保留为真实整片变化，不做对齐）。
    # 幅度低于局部材料变化，避免掩盖局部信号，但仍可在低噪声下被检出。
    global_offset = -0.02
    after_true = after_true + global_offset

    def make_repeats(truth, count, valid_template):
        cubes = []
        for r in range(count):
            values = truth + rng.normal(0, noise, truth.shape)
            values = np.clip(values, 0.01, 0.97)
            valid = valid_template.copy()
            cubes.append(create_cube(values.astype(np.float32), valid, band_rows,
                                     'reflectance', 'fraction', grid,
                                     {**meta, 'repeat_index': r}))
        return cubes

    # 共同缺测：边缘条带 + 内部逐波段缺测块
    valid_before = np.ones((height, width, bands), dtype=bool)
    valid_before[:3] = False
    valid_before[-3:] = False
    valid_before[:, :3] = False
    valid_before[:, -3:] = False
    valid_before[int(height * 0.15):int(height * 0.28),
                 int(width * 0.55):int(width * 0.66), 3:5] = False
    valid_after = valid_before.copy()
    # 处理后另一处缺测，制造部分像素两组共同波段更少
    valid_after[int(height * 0.55):int(height * 0.7),
                int(width * 0.08):int(width * 0.2), 4:] = False

    before_cubes = make_repeats(base, repeats, valid_before)
    after_cubes = make_repeats(after_true, repeats, valid_after)

    # 重复不足示范：把处理前与处理后各若干次重复在同一区域全部置为无效，使该区域
    # 两组都只剩 2 次有效重复（nu=2 < 波段数 6），Hotelling 不可识别，明确展示自由度不足未决。
    thin_region = (np.indices((height, width))[0] >= int(height * 0.8)) & \
                  (np.indices((height, width))[0] < int(height * 0.92)) & \
                  (np.indices((height, width))[1] >= int(width * 0.35)) & \
                  (np.indices((height, width))[1] < int(width * 0.55))
    thinned_repeats = max(0, repeats - 2)  # 每组仅保留 2 次有效
    for r in range(thinned_repeats):
        thin_valid = before_cubes[r].valid.copy()
        thin_valid[thin_region, :] = False
        before_cubes[r] = create_cube(before_cubes[r].values, thin_valid, band_rows,
                                      'reflectance', 'fraction', grid,
                                      {**meta, 'repeat_index': r, 'thinned_region': True})
        thin_valid_after = after_cubes[r].valid.copy()
        thin_valid_after[thin_region, :] = False
        after_cubes[r] = create_cube(after_cubes[r].values, thin_valid_after, band_rows,
                                     'reflectance', 'fraction', grid,
                                     {**meta, 'repeat_index': r, 'thinned_region': True})

    return {
        'before': before_cubes,
        'after': after_cubes,
        'ground_truth': {
            'local_change_pixels': int(change_mask.sum()),
            'local_change_spectral_offset': spectral_change.tolist(),
            'global_brightness_offset': global_offset,
            'noise_sigma': noise,
            'repeats_per_group': repeats,
        },
        'grid': grid,
        'bands': band_rows,
    }


def change_demo_description():
    return {
        'name': '两时点变化分析示范',
        'description': ('处理前/后各6次同网格、同波长、同单位的重复反射率采集，'
                        '包含一处长波抬升的局部材料变化、整片轻微变暗、边缘与内部缺测块，'
                        '以及一个重复变薄区域；全部程序合成。'),
        'ground_truth_features': [
            '椭圆异物区：处理后长波波段反射率抬升约0.14（正式取向应稳定检出）',
            '整片亮度位移：处理后整体 -0.02，不做亮度对齐；筛查取向更易捕捉这类弱整片变化',
            '边缘与内部逐波段缺测：缺测不当零，相关像素进入未决（no_overlap）',
            '重复变薄区域：展示有效重复/自由度不足以识别 Hotelling 检验时的未决',
        ],
        'limitations': ['合成数据，不代表真实材料或仪器', '噪声为独立高斯，不含系统漂移建模'],
    }
