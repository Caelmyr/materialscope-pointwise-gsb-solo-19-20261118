import io
import json
import struct
import zipfile
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

from . import contracts as c
from .model import create_cube
from .io import read_image


RAW_DTYPES = {'uint8': 'u1', 'uint16': 'u2', 'int16': 'i2', 'float32': 'f4', 'float64': 'f8'}


def raw_specification(options):
    options = c.object_value(options, '原始采集布局')
    width = c.integer(options.get('width'), '宽度', 1, 1024)
    height = c.integer(options.get('height'), '高度', 1, 1024)
    count = c.integer(options.get('band_count'), '波段数', 1, 64)
    if width * height * count > c.LIMITS['samples']:
        raise c.Invalid('原始采集样本数超过容量')
    dtype = c.choice(options.get('dtype', 'uint16'), '采样类型', list(RAW_DTYPES))
    endian = c.choice(options.get('byte_order', 'little'), '字节序', ['little', 'big'])
    layout = c.choice(options.get('interleave', 'BIL'), '排列', ['BIP', 'BIL', 'BSQ'])
    offset = c.integer(options.get('header_bytes', 0), '头部字节', 0, 1000000)
    row_padding = c.integer(options.get('row_padding_bytes', 0), '行末填充字节', 0, 100000)
    bands = c.band_metadata(options.get('bands', [{'name': f'B{i + 1}', 'wavelength_nm': None} for i in range(count)]), count)
    grid = c.grid_metadata(options.get('grid'))
    nodata = options.get('nodata')
    if nodata is not None:
        nodata = c.number(nodata, '无效值', -1e30, 1e30)
    return {
        'width': width,
        'height': height,
        'band_count': count,
        'dtype': dtype,
        'byte_order': endian,
        'interleave': layout,
        'header_bytes': offset,
        'row_padding_bytes': row_padding,
        'bands': bands,
        'grid': grid,
        'nodata': nodata,
        'unit': c.text(options.get('unit', 'DN'), '单位', 32),
    }


def decode_raw(data, options):
    specification = raw_specification(options)
    width = specification['width']
    height = specification['height']
    count = specification['band_count']
    prefix = '<' if specification['byte_order'] == 'little' else '>'
    dtype = np.dtype(prefix + RAW_DTYPES[specification['dtype']])
    layout = specification['interleave']
    row_values = width * count if layout in ('BIP', 'BIL') else width
    row_count = height if layout in ('BIP', 'BIL') else height * count
    row_bytes = row_values * dtype.itemsize
    stride = row_bytes + specification['row_padding_bytes']
    expected = specification['header_bytes'] + row_count * stride
    if len(data) != expected:
        raise c.Invalid(f'原始文件应为{expected}字节，实际为{len(data)}；请核对尺寸、头部和填充')
    offset = specification['header_bytes']
    rows = np.empty((row_count, row_values), dtype=dtype)
    view = memoryview(data)
    for row in range(row_count):
        begin = offset + row * stride
        rows[row] = np.frombuffer(view[begin:begin + row_bytes], dtype=dtype, count=row_values)
    if layout == 'BIP':
        values = rows.reshape(height, width, count)
    elif layout == 'BIL':
        values = rows.reshape(height, count, width).transpose(0, 2, 1)
    else:
        values = rows.reshape(count, height, width).transpose(1, 2, 0)
    valid = np.isfinite(values)
    if specification['nodata'] is not None:
        valid &= values != specification['nodata']
    report = {
        'source': 'raw_camera_capture',
        'specification': specification,
        'byte_count': len(data),
        'conversion': 'decoded native values converted to float32; invalid sentinel retained as mask',
    }
    cube = create_cube(values, valid, specification['bands'], 'raw', specification['unit'], specification['grid'], report)
    return cube, report


def safe_stack_zip(data):
    if len(data) > c.LIMITS['upload_bytes']:
        raise c.Invalid('波段包超过100MB')
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise c.Invalid('波段包不是ZIP') from None
    names = []
    expanded = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or '..' in path.parts or '\\' in info.filename:
            archive.close()
            raise c.Invalid('波段包含不安全路径')
        if info.filename in names or info.flag_bits & 1:
            archive.close()
            raise c.Invalid('波段包包含重复文件或加密条目')
        names.append(info.filename)
        expanded += info.file_size
        if len(names) > 130 or expanded > c.LIMITS['archive_bytes']:
            archive.close()
            raise c.Invalid('波段包展开超过容量')
    if 'bands.json' not in names:
        archive.close()
        raise c.Invalid('波段包缺少bands.json')
    return archive


def read_stack_manifest(value):
    value = c.object_value(value, '波段包说明')
    if value.get('schema') != 'materialscope_band_stack_v1':
        raise c.Invalid('不支持的波段包格式')
    entries = c.array_value(value.get('bands'), '波段列表', 1, 64)
    result = []
    seen = set()
    for index, entry in enumerate(entries):
        entry = c.object_value(entry, '波段条目')
        filename = c.text(entry.get('file'), '波段文件', 200)
        path = PurePosixPath(filename)
        if path.is_absolute() or '..' in path.parts or '\\' in filename:
            raise c.Invalid('波段文件路径无效')
        if filename in seen:
            raise c.Invalid('同一文件不能重复作为波段')
        seen.add(filename)
        channel = c.integer(entry.get('channel', 0), '图像通道', 0, 3)
        wave = entry.get('wavelength_nm')
        if wave is not None:
            wave = c.number(wave, '波长nm', 100, 30000)
        gain = c.number(entry.get('gain', 1), '解码增益', -1e6, 1e6)
        offset = c.number(entry.get('offset', 0), '解码偏移', -1e12, 1e12)
        nodata = entry.get('nodata')
        if nodata is not None:
            nodata = c.number(nodata, '无效值')
        result.append({
            'file': filename,
            'channel': channel,
            'name': c.text(entry.get('name', f'B{index + 1}'), '波段名称', 50),
            'wavelength_nm': wave,
            'fwhm_nm': entry.get('fwhm_nm'),
            'gain': gain,
            'offset': offset,
            'nodata': nodata,
        })
    return {
        'bands': result,
        'grid': c.grid_metadata(value.get('grid')),
        'domain': c.choice(value.get('domain', 'raw'), '数值域', ['raw', 'reflectance', 'derived']),
        'unit': c.text(value.get('unit', 'DN'), '单位', 32),
    }


def decode_band_image(data, channel):
    try:
        image = Image.open(io.BytesIO(data))
        if getattr(image, 'n_frames', 1) != 1:
            raise c.Invalid('波段文件必须是单页图像')
        if image.width > 1024 or image.height > 1024:
            raise c.Invalid('波段图像尺寸超过1024')
        image.load()
        if image.mode == 'P':
            image = image.convert('RGBA')
        values = np.asarray(image)
        if values.ndim == 2:
            if channel != 0:
                raise c.Invalid('单通道波段的channel必须为0')
            selected = values.astype(np.float64)
            valid = np.isfinite(selected)
        else:
            if channel >= values.shape[2]:
                raise c.Invalid('请求的图像通道不存在')
            selected = values[:, :, channel].astype(np.float64)
            valid = np.isfinite(selected)
            if values.shape[2] == 4 and channel < 3:
                valid &= values[:, :, 3] > 0
        return selected, valid
    except c.Invalid:
        raise
    except Exception as error:
        raise c.Invalid('波段图像不能解码：' + str(error)[:100]) from None


def decode_band_stack(data):
    with safe_stack_zip(data) as archive:
        try:
            manifest = read_stack_manifest(json.loads(archive.read('bands.json')))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise c.Invalid('bands.json不是有效JSON') from None
        values = []
        masks = []
        expected_shape = None
        for entry in manifest['bands']:
            if entry['file'] not in archive.namelist():
                raise c.Invalid('缺少波段文件：' + entry['file'])
            field, valid = decode_band_image(archive.read(entry['file']), entry['channel'])
            if expected_shape is None:
                expected_shape = field.shape
            elif expected_shape != field.shape:
                raise c.Invalid('波段图像尺寸不一致；导入阶段不隐式缩放')
            if entry['nodata'] is not None:
                valid &= field != entry['nodata']
            field = field * entry['gain'] + entry['offset']
            valid &= np.isfinite(field)
            values.append(field)
            masks.append(valid)
        bands = [{key: entry.get(key) for key in ('name', 'wavelength_nm', 'fwhm_nm')} for entry in manifest['bands']]
        report = {'source': 'band_stack', 'manifest': manifest, 'per_band_validity': True, 'no_implicit_registration': True}
        cube = create_cube(np.stack(values, axis=2), np.stack(masks, axis=2), bands, manifest['domain'], manifest['unit'], manifest['grid'], report)
        return cube, report


def acquisition_template():
    return {
        'schema': 'materialscope_band_stack_v1',
        'domain': 'raw',
        'unit': 'DN',
        'grid': {'x_step': 1, 'y_step': 1, 'origin': [0, 0], 'unit': 'pixel'},
        'bands': [
            {'file': 'band_450.tiff', 'name': '450 nm', 'wavelength_nm': 450, 'channel': 0, 'gain': 1, 'offset': 0},
            {'file': 'band_550.tiff', 'name': '550 nm', 'wavelength_nm': 550, 'channel': 0, 'gain': 1, 'offset': 0},
            {'file': 'band_650.tiff', 'name': '650 nm', 'wavelength_nm': 650, 'channel': 0, 'gain': 1, 'offset': 0},
        ],
    }
