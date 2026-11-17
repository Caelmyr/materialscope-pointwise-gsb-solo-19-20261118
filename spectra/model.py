from dataclasses import dataclass
import copy
import hashlib
import json

import numpy as np

from . import contracts as c


@dataclass(frozen=True)
class Cube:
    values: np.ndarray
    valid: np.ndarray
    bands: tuple
    domain: str
    unit: str
    grid: dict
    metadata: dict

    @property
    def shape(self):
        return self.values.shape

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    @property
    def count(self):
        return self.shape[2]

    def common_valid(self, bands=None):
        selected = list(range(self.count)) if bands is None else bands
        return np.all(self.valid[:, :, selected], axis=2)

    def any_valid(self):
        return np.any(self.valid, axis=2)

    def wavelengths(self, required=True):
        values = [item['wavelength_nm'] for item in self.bands]
        if required and any(item is None for item in values):
            raise c.Invalid('此操作需要每一波段具有物理波长')
        return np.asarray(values, dtype=float)

    def sorted_wavelengths(self):
        values = self.wavelengths()
        order = np.argsort(values, kind='stable')
        return values[order], order

    def samples(self, bands=None, maximum=50000, seed=71):
        selected = list(range(self.count)) if bands is None else bands
        valid = self.common_valid(selected)
        indices = np.flatnonzero(valid)
        if len(indices) < 2:
            raise c.Invalid('共同有效像素不足')
        if len(indices) > maximum:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(indices, maximum, replace=False))
        values = self.values[:, :, selected].reshape(-1, len(selected))
        return values[indices].astype(np.float64), indices

    def pixel(self, x, y):
        x = c.integer(x, '横坐标', 0, self.width - 1)
        y = c.integer(y, '纵坐标', 0, self.height - 1)
        valid = self.valid[y, x]
        return {
            'x': x,
            'y': y,
            'world': self.pixel_center(x, y),
            'values': [float(v) if ok else None for v, ok in zip(self.values[y, x], valid)],
            'valid': valid.tolist(),
            'bands': list(self.bands),
            'domain': self.domain,
            'unit': self.unit,
        }

    def pixel_center(self, x, y):
        return [
            self.grid['origin'][0] + (x + 0.5) * self.grid['x_step'],
            self.grid['origin'][1] + (y + 0.5) * self.grid['y_step'],
        ]

    def summary(self):
        counts = self.valid.sum(axis=(0, 1))
        return {
            'width': self.width,
            'height': self.height,
            'band_count': self.count,
            'bands': list(self.bands),
            'domain': self.domain,
            'unit': self.unit,
            'grid': copy.deepcopy(self.grid),
            'metadata': copy.deepcopy(self.metadata),
            'valid_counts': counts.tolist(),
            'common_valid_count': int(self.common_valid().sum()),
            'any_valid_count': int(self.any_valid().sum()),
            'bytes': int(self.values.nbytes + self.valid.nbytes),
        }

    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update(self.values.tobytes(order='C'))
        digest.update(self.valid.tobytes(order='C'))
        description = {
            'shape': self.shape,
            'bands': self.bands,
            'domain': self.domain,
            'unit': self.unit,
            'grid': self.grid,
            'metadata': self.metadata,
        }
        digest.update(json.dumps(description, sort_keys=True, ensure_ascii=False).encode())
        return digest.hexdigest()


def create_cube(values, valid=None, bands=None, domain='raw', unit='DN', grid=None, metadata=None):
    values = np.asarray(values)
    if values.dtype.kind not in 'fiu':
        raise c.Invalid('立方体必须是实数数组')
    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3:
        raise c.Invalid('立方体形状必须为高×宽×波段')
    height, width, count = values.shape
    if not 1 <= height <= c.LIMITS['height'] or not 1 <= width <= c.LIMITS['width']:
        raise c.Invalid('图像每边须为1–1024像素')
    if not 1 <= count <= c.LIMITS['bands']:
        raise c.Invalid('波段数须为1–64')
    if values.size > c.LIMITS['samples']:
        raise c.Invalid('立方体超过1600万个采样值')
    values = values.astype(np.float32)
    finite = np.isfinite(values)
    if valid is None:
        valid = finite
    else:
        valid = np.asarray(valid)
        if valid.dtype.kind not in 'biu':
            raise c.Invalid('有效域必须是布尔或整数数组')
        if valid.shape == (height, width):
            valid = np.repeat(valid[:, :, None], count, axis=2)
        if valid.shape != values.shape:
            raise c.Invalid('有效域应与立方体同形或为共同二维有效域')
        valid = valid.astype(bool) & finite
    if not np.any(valid):
        raise c.Invalid('立方体没有有效采样')
    values = np.ascontiguousarray(np.where(valid, values, 0), dtype=np.float32)
    valid = np.ascontiguousarray(valid, dtype=bool)
    if bands is None:
        bands = [{'name': f'B{i + 1}', 'wavelength_nm': None} for i in range(count)]
    bands = c.band_metadata(bands, count)
    domain = c.choice(domain, '数值域', ['raw', 'reflectance', 'derived', 'labels'])
    unit = c.text(unit, '数值单位', 32)
    grid = c.grid_metadata(grid)
    metadata = c.options_copy(metadata or {}, maximum=5000000)
    values.setflags(write=False)
    valid.setflags(write=False)
    return Cube(values, valid, tuple(bands), domain, unit, grid, metadata)


def derived_cube(source, values, valid, names, operation, report, unit='unitless', wavelengths=None, domain='derived'):
    if np.asarray(values).ndim == 2:
        values = np.asarray(values)[:, :, None]
    names = c.array_value(names, '结果波段名', np.asarray(values).shape[2], np.asarray(values).shape[2])
    wavelengths = wavelengths or [None] * len(names)
    bands = [{'name': name, 'wavelength_nm': wave} for name, wave in zip(names, wavelengths)]
    metadata = {
        'operation': operation,
        'report': c.json_safe(report),
        'source_fingerprint': source.fingerprint(),
        'source_domain': source.domain,
        'source_unit': source.unit,
    }
    return create_cube(values, valid, bands, domain, unit, source.grid, metadata)


def same_spectral_grid(first, second):
    if first.count != second.count:
        raise c.Invalid('参考影像波段数量不一致')
    first_wave = [item['wavelength_nm'] for item in first.bands]
    second_wave = [item['wavelength_nm'] for item in second.bands]
    if first_wave != second_wave:
        raise c.Invalid('参考影像波长和顺序不一致')


def same_spatial_grid(first, second):
    if first.shape[:2] != second.shape[:2] or first.grid != second.grid:
        raise c.Invalid('参考影像空间网格不一致')


def source_preserving(source, values, valid, operation, report):
    metadata = {
        'operation': operation,
        'report': c.json_safe(report),
        'source_fingerprint': source.fingerprint(),
        'source_domain': source.domain,
        'source_unit': source.unit,
    }
    return create_cube(values, valid, list(source.bands), source.domain, source.unit, source.grid, metadata)


def metadata_replacement(source, options):
    options = c.object_value(options, '元数据')
    bands = options.get('bands', list(source.bands))
    domain = options.get('domain', source.domain)
    unit = options.get('unit', source.unit)
    grid = options.get('grid', source.grid)
    metadata = copy.deepcopy(source.metadata)
    metadata['metadata_parent'] = source.fingerprint()
    metadata['metadata_note'] = c.text(options.get('note', '用户确认元数据'), '说明', 1000)
    return create_cube(source.values, source.valid, bands, domain, unit, grid, metadata)


def spectral_band(source, target, tolerance=100):
    wave = source.wavelengths()
    target = c.number(target, '目标波长', 100, 30000)
    nearest = int(np.argmin(np.abs(wave - target)))
    if abs(wave[nearest] - target) > tolerance:
        raise c.Invalid('没有足够接近目标波长的波段')
    return nearest
