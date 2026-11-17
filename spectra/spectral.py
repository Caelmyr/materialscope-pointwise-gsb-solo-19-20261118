import ast

import numpy as np
from scipy import ndimage

from . import contracts as c
from .model import source_preserving, derived_cube
from .numerics import safe_ratio


def smooth_spectra(source, options, references):
    waves, order = source.sorted_wavelengths()
    half_width = c.number(options.get('half_width_nm', 80), '半窗宽nm', 0.1, 10000)
    minimum_support = c.integer(options.get('minimum_support', 2), '最少支持波段', 1, source.count)
    preserve_missing = c.boolean(options.get('preserve_missing', True), '保留缺测')
    values = source.values[:, :, order].astype(float)
    valid = source.valid[:, :, order]
    output = np.zeros_like(values)
    output_valid = np.zeros_like(valid)
    support = []
    for index, center in enumerate(waves):
        distance = np.abs(waves - center)
        included = distance <= half_width
        weights = np.exp(-0.5 * (distance / max(half_width / 2, 1e-8)) ** 2) * included
        effective = valid * weights[None, None, :]
        denominator = effective.sum(axis=2)
        counts = np.sum(valid & included[None, None, :], axis=2)
        usable = (denominator > 1e-10) & (counts >= minimum_support)
        if preserve_missing:
            usable &= valid[:, :, index]
        output[:, :, index] = np.divide(np.sum(values * effective, axis=2), denominator, out=np.zeros(source.shape[:2]), where=usable)
        output_valid[:, :, index] = usable
        support.append(int(included.sum()))
    reverse = np.argsort(order)
    report = {'half_width_nm': half_width, 'minimum_support': minimum_support, 'preserve_missing': preserve_missing, 'window_support_sorted': support}
    return source_preserving(source, output[:, :, reverse], output_valid[:, :, reverse], 'spectral_smooth', report), report


def spectral_derivative(source, options, references):
    waves, order = source.sorted_wavelengths()
    if len(waves) < 3:
        raise c.Invalid('波长导数至少需要3波段')
    mode = c.choice(options.get('order', 1), '导数阶数', [1, 2])
    values = source.values[:, :, order].astype(float)
    valid = source.valid[:, :, order]
    left_step = waves[1:-1] - waves[:-2]
    right_step = waves[2:] - waves[1:-1]
    left_slope = (values[:, :, 1:-1] - values[:, :, :-2]) / left_step
    right_slope = (values[:, :, 2:] - values[:, :, 1:-1]) / right_step
    if mode == 1:
        middle = (right_step * left_slope + left_step * right_slope) / (left_step + right_step)
    else:
        middle = 2 * (right_slope - left_slope) / (left_step + right_step)
    middle_valid = valid[:, :, :-2] & valid[:, :, 1:-1] & valid[:, :, 2:]
    names = [f'd{mode} {wave:g}nm' for wave in waves[1:-1]]
    report = {'order': mode, 'wavelengths_nm': waves[1:-1].tolist(), 'support': 'three adjacent measured wavelengths; edge bands omitted'}
    unit = source.unit + '/nm' + ('^2' if mode == 2 else '')
    return derived_cube(source, middle, middle_valid, names, 'spectral_derivative', report, unit, waves[1:-1].tolist()), report


def normalize_spectra(source, options, references):
    method = c.choice(options.get('method', 'l2'), '光谱归一化', ['l2', 'sum', 'snv', 'reference_band'])
    minimum = c.integer(options.get('minimum_bands', min(3, source.count)), '最少有效带', 1, source.count)
    values = source.values.astype(float)
    valid = source.valid.copy()
    count = valid.sum(axis=2)
    usable = count >= minimum
    if method == 'l2':
        denominator = np.sqrt(np.sum(np.where(valid, values ** 2, 0), axis=2))
        output, denominator_valid = safe_ratio(values, denominator[:, :, None], valid)
        usable &= denominator_valid.any(axis=2)
    elif method == 'sum':
        denominator = np.sum(np.where(valid, values, 0), axis=2)
        output, denominator_valid = safe_ratio(values, denominator[:, :, None], valid)
        usable &= denominator_valid.any(axis=2)
    elif method == 'reference_band':
        band = c.integer(options.get('reference_band', 0), '参考波段', 0, source.count - 1)
        denominator = values[:, :, band]
        output, denominator_valid = safe_ratio(values, denominator[:, :, None], valid)
        usable &= source.valid[:, :, band] & denominator_valid.any(axis=2)
    else:
        center = np.divide(np.sum(np.where(valid, values, 0), axis=2), count, out=np.zeros(source.shape[:2]), where=count > 0)
        square = np.sum(np.where(valid, (values - center[:, :, None]) ** 2, 0), axis=2)
        deviation = np.sqrt(np.divide(square, count, out=np.zeros_like(square), where=count > 0))
        output, denominator_valid = safe_ratio(values - center[:, :, None], deviation[:, :, None], valid)
        usable &= denominator_valid.any(axis=2)
    valid &= usable[:, :, None]
    report = {'method': method, 'minimum_bands': minimum, 'valid_pixels': int(usable.sum()), 'note': '归一化结果为派生域，不再标为物理反射率'}
    names = [item['name'] for item in source.bands]
    waves = [item['wavelength_nm'] for item in source.bands]
    return derived_cube(source, output, valid, names, 'spectral_normalize', report, 'unitless', waves), report


def upper_continuum(wavelengths, values):
    hull = []
    for index in range(len(wavelengths)):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            cross = (wavelengths[second] - wavelengths[first]) * (values[index] - values[second]) - (values[second] - values[first]) * (wavelengths[index] - wavelengths[second])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(index)
    return np.interp(wavelengths, wavelengths[hull], values[hull]), hull


def continuum_remove(source, options, references):
    waves, order = source.sorted_wavelengths()
    low = c.number(options.get('low_nm', float(waves[0])), '起始波长', 100, 30000)
    high = c.number(options.get('high_nm', float(waves[-1])), '结束波长', low + 0.01, 30000)
    selected_order = order[(waves >= low) & (waves <= high)]
    selected_waves = source.wavelengths()[selected_order]
    if len(selected_order) < 3:
        raise c.Invalid('连续谱窗口至少3波段')
    valid = source.common_valid(selected_order)
    indices = np.flatnonzero(valid)
    if len(indices) > 100000:
        raise c.Invalid('连续谱逐像素包络最多10万完整光谱，请先裁切')
    samples = source.values[:, :, selected_order].reshape(-1, len(selected_order))
    output = np.zeros_like(samples)
    output_valid = np.zeros_like(samples, dtype=bool)
    depth = np.zeros(source.width * source.height, dtype=float)
    center = np.zeros_like(depth)
    area = np.zeros_like(depth)
    for pixel in indices:
        envelope, hull = upper_continuum(selected_waves, samples[pixel])
        usable = envelope > 1e-10
        ratio = np.divide(samples[pixel], envelope, out=np.zeros(len(selected_order)), where=usable)
        output[pixel] = ratio
        output_valid[pixel] = usable
        if usable.all():
            absorption = 1 - ratio
            deepest = int(np.argmax(absorption))
            depth[pixel] = absorption[deepest]
            center[pixel] = selected_waves[deepest]
            area[pixel] = np.trapezoid(absorption, selected_waves)
    report = {
        'low_nm': low,
        'high_nm': high,
        'complete_pixels': len(indices),
        'mean_absorption_depth': float(depth[indices].mean()) if len(indices) else None,
        'mean_absorption_area_nm': float(area[indices].mean()) if len(indices) else None,
        'envelope': 'upper piecewise linear hull on physical wavelengths',
    }
    if options.get('output', 'ratio') == 'features':
        field = np.stack([depth, center, area], axis=-1).reshape(source.height, source.width, 3)
        feature_valid = output_valid.all(axis=1).reshape(source.shape[:2])
        names = ['Absorption depth', 'Center wavelength nm', 'Integrated depth nm']
        return derived_cube(source, field, feature_valid, names, 'continuum_features', report, 'mixed'), report
    names = [f'CR {wave:g}nm' for wave in selected_waves]
    return derived_cube(source, output.reshape(source.height, source.width, -1), output_valid.reshape(source.height, source.width, -1), names, 'continuum_remove', report, 'ratio', selected_waves.tolist()), report


class Formula:
    functions = {'sqrt', 'abs', 'log', 'exp', 'minimum', 'maximum', 'clip'}

    def __init__(self, source, expression):
        self.source = source
        self.expression = c.text(expression, '波段表达式', 500)
        try:
            self.tree = ast.parse(self.expression, mode='eval')
        except SyntaxError:
            raise c.Invalid('表达式语法无效') from None
        if len(list(ast.walk(self.tree))) > 100:
            raise c.Invalid('表达式过于复杂')
        self.used = set()

    def evaluate(self):
        values, valid = self.node(self.tree.body)
        values = np.broadcast_to(values, self.source.shape[:2]).copy()
        valid = np.broadcast_to(valid, self.source.shape[:2]).copy()
        valid &= np.isfinite(values)
        if not self.used:
            raise c.Invalid('表达式必须引用至少一个波段，如b0')
        return values, valid, sorted(self.used)

    def node(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return c.number(node.value, '常量', -1e12, 1e12), True
        if isinstance(node, ast.Name) and node.id.startswith('b') and node.id[1:].isdigit():
            index = c.integer(node.id[1:], '波段索引', 0, self.source.count - 1)
            self.used.add(index)
            return self.source.values[:, :, index].astype(float), self.source.valid[:, :, index]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            values, valid = self.node(node.operand)
            return (-values if isinstance(node.op, ast.USub) else values), valid
        if isinstance(node, ast.BinOp):
            return self.binary(node)
        if isinstance(node, ast.Call):
            return self.call(node)
        raise c.Invalid('表达式只允许波段、数值和受限数学运算')

    def binary(self, node):
        left, left_valid = self.node(node.left)
        right, right_valid = self.node(node.right)
        valid = np.asarray(left_valid) & np.asarray(right_valid)
        with np.errstate(all='ignore'):
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                return safe_ratio(left, right, valid)
            elif isinstance(node.op, ast.Pow):
                if not np.isscalar(right) or not -8 <= right <= 8:
                    raise c.Invalid('幂指数必须是-8至8的常量')
                result = np.power(left, right)
            else:
                raise c.Invalid('不支持此运算符')
        return result, valid & np.isfinite(result)

    def call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in self.functions or node.keywords:
            raise c.Invalid('不支持此函数')
        name = node.func.id
        expected = 3 if name == 'clip' else 2 if name in ('minimum', 'maximum') else 1
        if len(node.args) != expected:
            raise c.Invalid('函数参数数量不正确')
        arguments = [self.node(item) for item in node.args]
        values = [item[0] for item in arguments]
        valid = np.asarray(True)
        for _, mask in arguments:
            valid = valid & np.asarray(mask)
        with np.errstate(all='ignore'):
            if name == 'sqrt':
                result = np.sqrt(values[0])
            elif name == 'abs':
                result = np.abs(values[0])
            elif name == 'log':
                result = np.log(values[0])
            elif name == 'exp':
                result = np.exp(np.clip(values[0], -80, 80))
            elif name == 'minimum':
                result = np.minimum(values[0], values[1])
            elif name == 'maximum':
                result = np.maximum(values[0], values[1])
            else:
                result = np.clip(values[0], values[1], values[2])
        return result, valid & np.isfinite(result)


def band_formula(source, options, references):
    expression = options.get('expression', '(b1-b0)/(b1+b0)')
    values, valid, used = Formula(source, expression).evaluate()
    name = c.text(options.get('name', 'Band formula'), '结果名', 60)
    unit = c.text(options.get('unit', 'unitless'), '结果单位', 32)
    report = {'expression': expression, 'used_bands': used, 'valid_count': int(valid.sum()), 'invalid_policy': 'intersection of operands plus arithmetic domain'}
    return derived_cube(source, values, valid, [name], 'band_formula', report, unit), report


def band_ratio(source, options, references):
    first = c.integer(options.get('first', 0), '第一波段', 0, source.count - 1)
    second = c.integer(options.get('second', min(1, source.count - 1)), '第二波段', 0, source.count - 1)
    mode = c.choice(options.get('mode', 'normalized_difference'), '指数方式', ['ratio', 'difference', 'normalized_difference'])
    a = source.values[:, :, first].astype(float)
    b = source.values[:, :, second].astype(float)
    valid = source.valid[:, :, first] & source.valid[:, :, second]
    if mode == 'ratio':
        output, valid = safe_ratio(a, b, valid)
    elif mode == 'difference':
        output = a - b
    else:
        output, valid = safe_ratio(a - b, a + b, valid)
    report = {'mode': mode, 'first': first, 'second': second, 'valid_count': int(valid.sum())}
    return derived_cube(source, output, valid, [mode], 'band_ratio', report, source.unit if mode == 'difference' else 'ratio'), report
