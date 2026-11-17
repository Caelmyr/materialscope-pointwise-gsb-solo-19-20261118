import csv
import io
import json
import zipfile
from pathlib import PurePosixPath

import numpy as np
from PIL import Image, ImageOps

from . import contracts as c
from .model import create_cube


def cube_bytes(cube):
    metadata = {
        'schema': 'materialscope_cube_v1',
        'bands': list(cube.bands),
        'domain': cube.domain,
        'unit': cube.unit,
        'grid': cube.grid,
        'metadata': cube.metadata,
    }
    output = io.BytesIO()
    np.savez_compressed(
        output,
        cube=cube.values,
        valid=cube.valid,
        description=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return output.getvalue()


def inspect_npz(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            names = [item.filename for item in entries]
            if len(names) != len(set(names)):
                raise c.Invalid('NPZ含重复条目')
            if len(entries) > 12:
                raise c.Invalid('NPZ条目过多')
            if sum(item.file_size for item in entries) > c.LIMITS['archive_bytes']:
                raise c.Invalid('NPZ展开超过容量')
            for item in entries:
                if item.filename not in ('cube.npy', 'valid.npy', 'description.npy', 'wavelengths.npy'):
                    raise c.Invalid('NPZ含未知条目：' + item.filename)
                if item.flag_bits & 1:
                    raise c.Invalid('不支持加密NPZ')
    except zipfile.BadZipFile:
        raise c.Invalid('不是有效的NPZ文件') from None


def read_npz(data):
    inspect_npz(data)
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if 'cube' not in archive:
                raise c.Invalid('NPZ缺少cube数组')
            values = archive['cube']
            valid = archive['valid'] if 'valid' in archive else None
            description = {}
            if 'description' in archive:
                text_value = archive['description']
                if text_value.shape != () or text_value.dtype.kind not in 'US':
                    raise c.Invalid('description必须是JSON字符串标量')
                if len(str(text_value.item())) > 5000000:
                    raise c.Invalid('description过长')
                description = json.loads(str(text_value.item()))
                c.object_value(description, 'description')
            if 'wavelengths' in archive and 'bands' not in description:
                waves = c.finite_array(archive['wavelengths'], '波长', 1, 64)
                description['bands'] = [
                    {'name': f'{wave:g} nm', 'wavelength_nm': float(wave)}
                    for wave in waves
                ]
            return create_cube(
                values,
                valid,
                description.get('bands'),
                description.get('domain', 'raw'),
                description.get('unit', 'DN'),
                description.get('grid'),
                description.get('metadata', {}),
            )
    except c.Invalid:
        raise
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as error:
        raise c.Invalid('无法读取NPZ：' + str(error)[:160]) from None


def read_image(data, name):
    try:
        image = Image.open(io.BytesIO(data))
        if getattr(image, 'n_frames', 1) != 1:
            raise c.Invalid('多页图像请转换为多波段NPZ后导入')
        if image.width > c.LIMITS['width'] or image.height > c.LIMITS['height']:
            raise c.Invalid('图像边长超过1024')
        image = ImageOps.exif_transpose(image)
        image.load()
        if image.mode in ('I', 'F', 'I;16', 'I;16L', 'I;16B'):
            values = np.asarray(image).astype(np.float32)
            return create_cube(values, domain='raw', unit='DN', metadata={'input_name': name})
        if image.mode in ('1', 'L'):
            values = np.asarray(image.convert('L'), dtype=np.float32) / 255
            return create_cube(values, domain='raw', unit='normalized_DN', metadata={'input_name': name})
        rgba = np.asarray(image.convert('RGBA'), dtype=np.uint8)
        values = rgba[:, :, :3].astype(np.float32) / 255
        valid = rgba[:, :, 3] > 0
        bands = [
            {'name': 'Red', 'wavelength_nm': None},
            {'name': 'Green', 'wavelength_nm': None},
            {'name': 'Blue', 'wavelength_nm': None},
        ]
        return create_cube(values, valid, bands, 'raw', 'normalized_DN', metadata={'input_name': name})
    except c.Invalid:
        raise
    except Exception as error:
        raise c.Invalid('无法解码图像：' + str(error)[:120]) from None


def import_cube(data, name):
    name = c.filename(name)
    if len(data) > c.LIMITS['upload_bytes']:
        raise c.Invalid('上传文件超过100MB')
    if name.lower().endswith('.npz'):
        return read_npz(data)
    if name.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg')):
        return read_image(data, name)
    raise c.Invalid('支持NPZ、PNG、JPEG和单页TIFF')


def band_tiff(cube, band):
    band = c.integer(band, '波段', 0, cube.count - 1)
    values = cube.values[:, :, band].copy()
    values[~cube.valid[:, :, band]] = np.nan
    image = Image.fromarray(values.astype(np.float32))
    output = io.BytesIO()
    image.save(output, format='TIFF', compression='tiff_deflate')
    return output.getvalue()


def mask_png(cube, band=None):
    if band is None:
        valid = cube.common_valid()
    else:
        band = c.integer(band, '波段', 0, cube.count - 1)
        valid = cube.valid[:, :, band]
    output = io.BytesIO()
    Image.fromarray(valid.astype(np.uint8) * 255).save(output, format='PNG')
    return output.getvalue()


def csv_bytes(rows, columns=None):
    output = io.StringIO(newline='')
    if columns is None:
        columns = list(rows[0]) if rows else []
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        converted = {}
        for key in columns:
            value = row.get(key)
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(c.json_safe(value), ensure_ascii=False)
            if isinstance(value, str) and value.startswith(('=', '+', '-', '@')):
                value = "'" + value
            converted[key] = value
        writer.writerow(converted)
    return ('\ufeff' + output.getvalue()).encode('utf-8')


def spectrum_csv(sample):
    rows = []
    for index, band in enumerate(sample['bands']):
        rows.append({
            'index': index,
            'name': band['name'],
            'wavelength_nm': band.get('wavelength_nm'),
            'value': sample['values'][index],
            'valid': sample['valid'][index],
            'unit': sample['unit'],
        })
    return csv_bytes(rows)


def read_signature_csv(data):
    if len(data) > 1000000:
        raise c.Invalid('光谱CSV超过1MB')
    try:
        text_value = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise c.Invalid('光谱CSV须为UTF-8') from None
    reader = csv.DictReader(io.StringIO(text_value))
    required = {'wavelength_nm', 'value'}
    if not required.issubset(reader.fieldnames or []):
        raise c.Invalid('CSV须含wavelength_nm和value列')
    waves = []
    values = []
    for row in reader:
        if len(waves) >= 512:
            raise c.Invalid('参考光谱最多512个采样点')
        waves.append(c.number(row['wavelength_nm'], '波长', 100, 30000))
        values.append(c.number(row['value'], '光谱数值', -1e12, 1e12))
    if len(waves) < 2 or len(set(waves)) != len(waves):
        raise c.Invalid('光谱至少两个互不重复的波长')
    order = np.argsort(waves)
    return {'wavelengths_nm': np.asarray(waves)[order].tolist(), 'values': np.asarray(values)[order].tolist()}


def safe_archive_entries(data):
    if len(data) > c.LIMITS['archive_bytes']:
        raise c.Invalid('工程包超过120MB')
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise c.Invalid('工程包不是有效ZIP') from None
    infos = archive.infolist()
    names = set()
    total = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or '..' in path.parts or '\\' in info.filename:
            archive.close()
            raise c.Invalid('工程包包含不安全路径')
        if info.filename in names:
            archive.close()
            raise c.Invalid('工程包含重复文件')
        names.add(info.filename)
        total += info.file_size
        if total > c.LIMITS['archive_bytes'] or len(names) > 150:
            archive.close()
            raise c.Invalid('工程包展开超过容量')
        if info.flag_bits & 1:
            archive.close()
            raise c.Invalid('不支持加密工程包')
    if 'project.json' not in names:
        archive.close()
        raise c.Invalid('工程包缺少project.json')
    return archive


def result_bundle(cube, name, report):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('cube.npz', cube_bytes(cube))
        archive.writestr('common-valid.png', mask_png(cube))
        archive.writestr('description.json', json.dumps({
            'name': name,
            'summary': cube.summary(),
            'report': report,
            'fingerprint': cube.fingerprint(),
            'coordinate_convention': 'pixel centers are origin + (index + 0.5) * step',
            'invalid_semantics': 'per-band mask is authoritative; stored zero is not an observation',
        }, ensure_ascii=False, indent=2))
        for index in range(cube.count):
            archive.writestr(f'bands/{index:02d}.tiff', band_tiff(cube, index))
    return output.getvalue()
