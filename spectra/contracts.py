import math
import re
import uuid
from datetime import datetime, timezone

import numpy as np


class Invalid(ValueError):
    pass


class Conflict(Invalid):
    pass


LIMITS = {
    'width': 1024,
    'height': 1024,
    'bands': 64,
    'samples': 16000000,
    'assets': 100,
    'regions': 100,
    'signatures': 100,
    'classifiers': 20,
    'archive_bytes': 120000000,
    'upload_bytes': 100000000,
    'history': 50,
}


def object_value(value, label):
    if not isinstance(value, dict):
        raise Invalid(label + '必须是对象')
    return value


def array_value(value, label, minimum=0, maximum=1000):
    if not isinstance(value, (list, tuple)):
        raise Invalid(label + '必须是数组')
    if not minimum <= len(value) <= maximum:
        raise Invalid(f'{label}长度须为{minimum}–{maximum}')
    return list(value)


def number(value, label, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise Invalid(label + '不能是布尔值')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise Invalid(label + '必须是有限数值') from None
    if not math.isfinite(result):
        raise Invalid(label + '必须是有限数值')
    if minimum is not None and result < minimum:
        raise Invalid(f'{label}不能小于{minimum}')
    if maximum is not None and result > maximum:
        raise Invalid(f'{label}不能大于{maximum}')
    return result


def integer(value, label, minimum=None, maximum=None):
    result = number(value, label, minimum, maximum)
    if result != math.floor(result):
        raise Invalid(label + '必须是整数')
    return int(result)


def text(value, label, maximum=120, allow_empty=False):
    if not isinstance(value, str):
        raise Invalid(label + '必须是文字')
    result = value.strip()
    if not allow_empty and not result:
        raise Invalid(label + '不能为空')
    if len(result) > maximum:
        raise Invalid(f'{label}不能超过{maximum}字')
    if any(ord(char) < 32 and char not in '\n\t' for char in result):
        raise Invalid(label + '含控制字符')
    return result


def choice(value, label, options):
    if value not in options:
        raise Invalid(label + '须为：' + ' / '.join(map(str, options)))
    return value


def boolean(value, label):
    if not isinstance(value, bool):
        raise Invalid(label + '必须为布尔值')
    return value


def identifier(value, label='标识'):
    result = text(value, label, 64)
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', result):
        raise Invalid(label + '只能含字母、数字、横线和下划线')
    return result


def generated_id(prefix):
    return prefix + '_' + uuid.uuid4().hex[:16]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def finite_array(value, label, dimensions=None, maximum=16000000):
    try:
        result = np.asarray(value)
    except (ValueError, TypeError):
        raise Invalid(label + '不是规则数值数组') from None
    if result.dtype.kind not in 'fiu':
        raise Invalid(label + '必须是实数数组')
    if result.size > maximum:
        raise Invalid(label + '超过数组容量')
    if dimensions is not None and result.ndim != dimensions:
        raise Invalid(f'{label}必须是{dimensions}维')
    output = result.astype(np.float64)
    if not np.isfinite(output).all():
        raise Invalid(label + '不能含非有限数')
    return output


def index_list(value, length, label='波段', minimum=1):
    values = array_value(value, label, minimum, length)
    result = [integer(item, label, 0, length - 1) for item in values]
    if len(set(result)) != len(result):
        raise Invalid(label + '不能重复')
    return result


def pair(value, label, minimum=None, maximum=None):
    values = array_value(value, label, 2, 2)
    return [number(item, label, minimum, maximum) for item in values]


def color(value):
    result = text(value, '颜色', 7)
    if not re.fullmatch(r'#[0-9a-fA-F]{6}', result):
        raise Invalid('颜色须为#RRGGBB')
    return result.lower()


def revision(value):
    return integer(value, '修订号', 0)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise Invalid('结果中存在不可序列化值')


def bounds(value, width, height):
    values = array_value(value, '矩形', 4, 4)
    x0 = integer(values[0], '左边界', 0, width - 1)
    y0 = integer(values[1], '上边界', 0, height - 1)
    x1 = integer(values[2], '右边界', x0 + 1, width)
    y1 = integer(values[3], '下边界', y0 + 1, height)
    return x0, y0, x1, y1


def odd_window(value, label='窗口', maximum=31):
    result = integer(value, label, 1, maximum)
    if result % 2 != 1:
        raise Invalid(label + '必须为奇数')
    return result


def percent_range(low, high):
    a = number(low, '下分位', 0, 99.9)
    b = number(high, '上分位', 0.1, 100)
    if a >= b:
        raise Invalid('下分位必须小于上分位')
    return a, b


def filename(value, suffix=None):
    result = text(value, '文件名', 200)
    if '/' in result or '\\' in result or result in ('.', '..'):
        raise Invalid('文件名不能含目录')
    if suffix and not result.lower().endswith(suffix):
        raise Invalid('文件扩展名须为' + suffix)
    return result


def grid_metadata(value):
    value = object_value(value or {}, '空间刻度')
    sx = number(value.get('x_step', 1), '横向步长', 0.000001, 1000000)
    sy = number(value.get('y_step', 1), '纵向步长', 0.000001, 1000000)
    origin = pair(value.get('origin', [0, 0]), '空间原点', -1e9, 1e9)
    unit = choice(value.get('unit', 'pixel'), '长度单位', ['pixel', 'mm', 'um'])
    return {'x_step': sx, 'y_step': sy, 'origin': origin, 'unit': unit}


def band_metadata(value, count):
    rows = array_value(value, '波段信息', count, count)
    result = []
    names = set()
    for position, row in enumerate(rows):
        row = object_value(row, '波段信息')
        name = text(row.get('name', f'B{position + 1}'), '波段名称', 50)
        if name in names:
            raise Invalid('波段名称不能重复')
        names.add(name)
        wavelength = row.get('wavelength_nm')
        if wavelength is not None:
            wavelength = number(wavelength, '波长nm', 100, 30000)
        width = row.get('fwhm_nm')
        if width is not None:
            width = number(width, '带宽nm', 0.01, 10000)
        result.append({'name': name, 'wavelength_nm': wavelength, 'fwhm_nm': width})
    present = [row['wavelength_nm'] for row in result if row['wavelength_nm'] is not None]
    if len(set(present)) != len(present):
        raise Invalid('波长中心不能重复')
    return result


def options_copy(value, maximum=50000):
    import json
    value = object_value(value, '处理参数')
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(encoded) > maximum:
        raise Invalid('处理参数过大')
    return json.loads(encoded)
