import numpy as np
from scipy import ndimage

from .model import create_cube
from . import contracts as c


def material_signatures(wavelengths):
    x = np.asarray(wavelengths, dtype=float)
    polymer = 0.35 + 0.3 / (1 + np.exp(-(x - 680) / 75)) - 0.18 * np.exp(-((x - 940) / 45) ** 2)
    fiber = 0.5 + 0.05 * np.sin(x / 140) - 0.23 * np.exp(-((x - 760) / 75) ** 2)
    coating = 0.18 + 0.45 * np.exp(-((x - 590) / 160) ** 2) + 0.07 * (x - 400) / 600
    contaminant = 0.12 + 0.2 * (x - 400) / 600 + 0.13 * np.exp(-((x - 840) / 35) ** 2)
    return np.stack([polymer, fiber, coating, contaminant], axis=0)


def create_demo(width=192, height=144, bands=12, seed=71):
    width = c.integer(width, '演示宽度', 64, 512)
    height = c.integer(height, '演示高度', 64, 512)
    bands = c.integer(bands, '演示波段数', 6, 32)
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(430, 980, bands)
    signature_values = material_signatures(wavelengths)
    yy, xx = np.indices((height, width), dtype=float)
    nx = xx / width
    ny = yy / height
    labels = np.zeros((height, width), dtype=int)
    labels[(nx > 0.35) & (nx < 0.68)] = 1
    labels[nx >= 0.68] = 2
    labels[((nx - 0.58) ** 2 / 0.045 ** 2 + (ny - 0.48) ** 2 / 0.065 ** 2) < 1] = 3
    labels[((nx - 0.23) ** 2 / 0.035 ** 2 + (ny - 0.75) ** 2 / 0.04 ** 2) < 1] = 3
    reflectance = signature_values[labels]
    texture = 0.015 * np.sin(xx / 2.4) * np.cos(yy / 3.7)
    reflectance += texture[:, :, None]
    reflectance += 0.012 * rng.normal(size=reflectance.shape)
    reflectance = np.clip(reflectance, 0.01, 0.95)
    illumination = 0.75 + 0.18 * nx + 0.08 * ny + 0.03 * np.cos(nx * 8)
    spectral_power = 2400 + 900 * np.sin((wavelengths - 400) / 600 * np.pi)
    dark = 90 + 12 * nx[:, :, None] + 7 * ny[:, :, None] + 3 * np.arange(bands)[None, None, :]
    dark = np.broadcast_to(dark, reflectance.shape).copy()
    white = dark + illumination[:, :, None] * spectral_power
    raw = dark + reflectance / 0.99 * (white - dark)
    raw += rng.normal(0, 3, raw.shape)
    dark += rng.normal(0, 1, dark.shape)
    white += rng.normal(0, 2, white.shape)
    valid = np.ones(raw.shape, dtype=bool)
    valid[:4] = False
    valid[-4:] = False
    valid[:, :4] = False
    valid[:, -4:] = False
    valid[int(height * 0.2):int(height * 0.3), int(width * 0.48):int(width * 0.55), 2:4] = False
    band_rows = [{'name': f'{wave:.0f} nm', 'wavelength_nm': float(wave), 'fwhm_nm': 20.0} for wave in wavelengths]
    grid = {'x_step': 0.08, 'y_step': 0.08, 'origin': [0, 0], 'unit': 'mm'}
    metadata = {'synthetic': True, 'generator': 'material coupon fixture v1', 'seed': seed, 'not_real_observation': True}
    cubes = {
        'raw': create_cube(raw, valid, band_rows, 'raw', 'DN', grid, {**metadata, 'role': 'sample'}),
        'dark': create_cube(dark, valid, band_rows, 'raw', 'DN', grid, {**metadata, 'role': 'dark_reference'}),
        'white': create_cube(white, valid, band_rows, 'raw', 'DN', grid, {**metadata, 'role': 'white_reference', 'white_reflectance': 0.99}),
        'reflectance': create_cube(reflectance, valid, band_rows, 'reflectance', 'fraction', grid, {**metadata, 'role': 'synthetic_reflectance'}),
    }
    shifted = reflectance.copy()
    shifted_valid = valid.copy()
    shifts = []
    local_warps = []
    warp_yy, warp_xx = np.indices((height, width), dtype=float)
    for index in range(bands):
        dx = float(np.sin(index * 0.8) * 2.5)
        dy = float(np.cos(index * 0.6) * 1.8)
        # 局部位移场：随空间位置变化的拉伸/压缩，长波更明显；参考带本身为零
        stretch_scale = (index - (bands - 1) / 2) / max(bands - 1, 1)
        local_dx = stretch_scale * 2.2 * np.sin(warp_yy / max(height, 1) * np.pi)
        local_dy = stretch_scale * 1.4 * np.sin(warp_xx / max(width, 1) * np.pi)
        local_warps.append({'band_index': index, 'scale': round(stretch_scale, 4),
                            'max_local_dx': round(float(np.max(np.abs(local_dx))), 4),
                            'max_local_dy': round(float(np.max(np.abs(local_dy))), 4)})
        sample_x = warp_xx - dx - local_dx
        sample_y = warp_yy - dy - local_dy
        inside = (sample_x >= 0) & (sample_x <= width - 1) & (sample_y >= 0) & (sample_y <= height - 1)
        x0 = np.clip(np.floor(sample_x).astype(int), 0, width - 2)
        y0 = np.clip(np.floor(sample_y).astype(int), 0, height - 2)
        fx = np.clip(sample_x - x0, 0, 1)
        fy = np.clip(sample_y - y0, 0, 1)
        x1 = x0 + 1
        y1 = y0 + 1
        source = reflectance[:, :, index]
        warped = (source[y0, x0] * (1 - fx) * (1 - fy) + source[y0, x1] * fx * (1 - fy)
                  + source[y1, x0] * (1 - fx) * fy + source[y1, x1] * fx * fy)
        source_valid = valid[:, :, index].astype(float)
        warped_valid = (source_valid[y0, x0] * (1 - fx) * (1 - fy) + source_valid[y0, x1] * fx * (1 - fy)
                        + source_valid[y1, x0] * (1 - fx) * fy + source_valid[y1, x1] * fx * fy)
        shifted[:, :, index] = np.where(inside, warped, 0)
        shifted_valid[:, :, index] = inside & (warped_valid >= 0.99999)
        shifts.append([dx, dy])
    cubes['misregistered'] = create_cube(
        shifted, shifted_valid, band_rows, 'reflectance', 'fraction', grid,
        {**metadata, 'role': 'shifted_bands', 'synthetic_applied_shifts': shifts,
         'synthetic_local_warps': local_warps,
         'synthetic_note': '每带含整体平移与随空间变化的局部拉伸，用于演示可审阅局部配准'},
    )
    names = ['Polymer reference', 'Fiber reference', 'Coating reference']
    signatures = []
    for index, name in enumerate(names):
        signatures.append({
            'id': c.generated_id('signature'),
            'name': name,
            'wavelengths_nm': wavelengths.tolist(),
            'values': signature_values[index].tolist(),
            'std': None,
            'domain': 'reflectance',
            'unit': 'fraction',
            'source_fingerprint': None,
            'source_region': None,
            'source_counts': None,
            'created_at': c.timestamp(),
            'synthetic': True,
        })
    region_specs = [
        {'name': 'Polymer coupon', 'bounds': [int(width * 0.08), int(height * 0.15), int(width * 0.28), int(height * 0.5)], 'color': '#f4b860'},
        {'name': 'Fiber coupon', 'bounds': [int(width * 0.4), int(height * 0.6), int(width * 0.61), int(height * 0.86)], 'color': '#61d6bc'},
        {'name': 'Coating coupon', 'bounds': [int(width * 0.75), int(height * 0.2), int(width * 0.92), int(height * 0.55)], 'color': '#9aafff'},
    ]
    return cubes, signatures, region_specs


def demo_description():
    return {
        'name': '合成材料试片',
        'description': '三个材料区、两处异物、逐波段缺测块、空间照明渐变及一套错位波段影像；全部为程序合成。',
        'default_shape': [144, 192, 12],
        'wavelength_range_nm': [430, 980],
        'roles': ['raw sample', 'dark reference', 'white reference', 'reflectance fixture', 'shifted fixture'],
        'normal_workflows': [
            '选择原始试片，指定暗场和白板进行反射率标定',
            '选择反射率试片，查看分区光谱和材料谱库匹配',
            '对反射率试片做主成分、聚类与异常图，再分割异常对象',
            '对错位波段影像执行全局平移配准，或在局部配准审阅中确认对应点、圈定独立材料区域后重算局部变形',
            '导出工程，重启后重新打开或导入工程副本',
        ],
        'limitations': ['材料曲线是合成函数，不代表真实材料数据库', '标定示例未包含真实传感器非线性', '不存在自动深度或真实化学成分解释'],
    }
