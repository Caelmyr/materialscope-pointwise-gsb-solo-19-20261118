"""可审阅的局部波段配准。

设计边界（与产品约束一致）：

* 参考波段与每个待配波段之间由人工确认的对应点驱动；未确认的候选不约束结果。
* 用户圈定的材料区域各自独立求解三角剖分分段线性变形，区域之间不互相牵动。
* 只用分段仿射（Delaunay 重心坐标）映射；任何三角形出现折叠或翻转即报冲突，
  不会为了贴齐而折叠或翻转局部图像。
* 缺测、越界以及无法可靠确定的位置不会变成有效测量：严格双线性支持之外的像素
  在输出有效域中置为无效；未决波段默认整带不宣称配准（mask），也可显式选择
  保留原始数值并明确标记。
* 不同波长下纹理消失或重复条纹造成的位置歧义由归一化互相关候选显式呈现，
  不以单一高相关分数掩盖。
"""

import math

import numpy as np
from scipy.spatial import Delaunay

from . import contracts as c
from .numerics import bilinear_sample
from .registration import phase_translation


SESSION_LIMIT_POINTS = 400
SESSION_LIMIT_REGIONS = 20
SESSION_LIMIT_CANDIDATES = 12
MIN_TEMPLATE_SIDE = 5
MAX_TEMPLATE_SIDE = 64
MIN_TEMPLATE_VALID = 16


def _bounded(value, low, high, label):
    result = c.number(value, label, low, high)
    if abs(result - round(result)) < 1e-9:
        return float(round(result))
    return result


def validate_xy(value, width, height, label):
    pair = c.array_value(value, label, 2, 2)
    x = _bounded(pair[0], -0.5, width - 0.5, label + '横坐标')
    y = _bounded(pair[1], -0.5, height - 0.5, label + '纵坐标')
    return [x, y]


def validate_point(point, width, height):
    c.object_value(point, '对应点')
    pid = c.identifier(point['id'], '对应点标识') if point.get('id') else None
    band = c.integer(point.get('band'), '对应点波段', 0, None)
    reference = validate_xy(point.get('reference'), width, height, '参考点')
    moving = validate_xy(point.get('moving'), width, height, '待配点')
    status = c.choice(point.get('status', 'confirmed'), '对应点状态', ['confirmed', 'pending'])
    note = c.text(point.get('note', ''), '对应点说明', 200, True)
    candidates = point.get('candidates', [])
    rows = c.array_value(candidates, '候选位置', 0, SESSION_LIMIT_CANDIDATES)
    clean = []
    for row in rows:
        c.object_value(row, '候选位置')
        clean.append({
            'moving': validate_xy(row.get('moving'), width, height, '候选坐标'),
            'score': c.number(row.get('score'), '候选相关分', -1, 1),
            'ambiguous': c.boolean(row.get('ambiguous', False), '歧义标记'),
            'reason': c.text(row.get('reason', ''), '候选依据', 200, True),
        })
    return {
        'id': pid,
        'band': band,
        'reference': reference,
        'moving': moving,
        'status': status,
        'note': note,
        'candidates': clean,
    }

def validate_region(region, width, height):
    c.object_value(region, '材料区域')
    rid = c.identifier(region['id'], '区域标识') if region.get('id') else None
    name = c.text(region.get('name', '材料区域'), '区域名称')
    kind = c.choice(region.get('kind', 'polygon'), '区域形状', ['rectangle', 'ellipse', 'polygon'])
    color = c.color(region.get('color', '#67dbc1'))
    if kind == 'polygon':
        vertices = c.array_value(region.get('vertices'), '区域顶点', 3, 200)
        clean = [validate_xy(item, width, height, '区域顶点') for item in vertices]
    else:
        bounds = c.bounds(region.get('shape') or region.get('bounds'), width, height)
        clean = list(bounds)
    return {'id': rid, 'name': name, 'kind': kind, 'shape': clean, 'color': color}


def rasterize_region(region, height, width):
    yy, xx = np.indices((height, width), dtype=float)
    shape = region['shape']
    if region['kind'] == 'rectangle':
        x0, y0, x1, y1 = shape
        return (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    if region['kind'] == 'ellipse':
        x0, y0, x1, y1 = shape
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        rx = max((x1 - x0) / 2, 1e-9)
        ry = max((y1 - y0) / 2, 1e-9)
        return ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1
    return point_in_polygon(xx + 0.5, yy + 0.5, np.asarray(shape, dtype=float))


def point_in_polygon(x, y, polygon):
    inside = np.zeros(x.shape, dtype=bool)
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        crosses = ((y1 > y) != (y2 > y))
        x_cross = np.where(crosses, x1 + (y - y1) * (x2 - x1) / np.where(y2 == y1, 1e-12, y2 - y1), np.inf)
        inside ^= crosses & (x < x_cross)
    return inside


def validate_session_base(session, width, height, band_count):
    c.object_value(session, '局部配准审阅')
    reference_band = c.integer(session.get('reference_band', band_count // 2), '参考波段', 0, band_count - 1)
    max_displacement = c.number(session.get('max_displacement', 20), '最大局部位移像素', 0.5, min(width, height) / 2)
    unconstrained_policy = c.choice(
        session.get('unconstrained_policy', 'mask'), '无约束位置处理', ['mask', 'passthrough'],
    )
    points = [validate_point(item, width, height) for item in c.array_value(session.get('points', []), '对应点', 0, SESSION_LIMIT_POINTS)]
    regions = [validate_region(item, width, height) for item in c.array_value(session.get('regions', []), '材料区域', 0, SESSION_LIMIT_REGIONS)]
    for point in points:
        if not 0 <= point['band'] < band_count:
            raise c.Invalid('对应点波段超出影像范围')
        if point['band'] == reference_band:
            raise c.Invalid('参考波段不需要自身对应点')
    return reference_band, max_displacement, unconstrained_policy, points, regions


def _duplicate_pairs(points):
    seen = set()
    duplicates = []
    for point in points:
        key = (point['band'], round(point['reference'][0], 3), round(point['reference'][1], 3))
        if key in seen:
            duplicates.append(point['id'])
        seen.add(key)
    return duplicates

def region_conflicts(regions, height, width):
    conflicts = []
    masks = []
    for region in regions:
        mask = rasterize_region(region, height, width)
        if not mask.any():
            conflicts.append({
                'kind': 'empty_region',
                'severity': 'blocking',
                'region_id': region['id'],
                'message': f"区域{region['name']}未覆盖任何像素中心",
            })
        masks.append(mask)
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            overlap = int(np.logical_and(masks[i], masks[j]).sum())
            if overlap:
                conflicts.append({
                    'kind': 'region_overlap',
                    'severity': 'blocking',
                    'region_ids': [regions[i]['id'], regions[j]['id']],
                    'overlap_pixels': overlap,
                    'message': f"区域{regions[i]['name']}与{regions[j]['name']}重叠{overlap}像素，独立区域不能共享像素",
                })
    return conflicts, masks


def detect_conflicts(session, cube):
    height, width = cube.shape[:2]
    reference_band, max_displacement, unconstrained_policy, points, regions = validate_session_base(
        session, width, height, cube.count,
    )
    conflicts = []
    duplicates = _duplicate_pairs(points)
    for pid in duplicates:
        conflicts.append({'kind': 'duplicate_point', 'severity': 'blocking', 'point_id': pid,
                          'message': '同一波段在同一参考位置存在多个对应点，无法同时约束'})
    region_rows, region_masks = region_conflicts(regions, height, width)
    conflicts.extend(region_rows)
    region_of_pixel = np.full((height, width), -1, dtype=int)
    for index, mask in enumerate(region_masks):
        region_of_pixel[mask] = index

    per_band = {}
    for band in range(cube.count):
        if band == reference_band:
            continue
        band_points = [p for p in points if p['band'] == band and p['status'] == 'confirmed']
        displacements = [math.hypot(p['moving'][0] - p['reference'][0], p['moving'][1] - p['reference'][1]) for p in band_points]
        for point, displacement in zip(band_points, displacements):
            if displacement > max_displacement:
                conflicts.append({
                    'kind': 'displacement_limit',
                    'severity': 'blocking',
                    'point_id': point['id'],
                    'band': band,
                    'displacement_px': round(displacement, 3),
                    'message': f"对应点{point['id']}位移{displacement:.2f}像素，超过最大局部位移{max_displacement:g}",
                })
        groups = [[] for _ in regions]
        global_points = []
        for point in band_points:
            owner = region_of_pixel[int(round(point['reference'][1])), int(round(point['reference'][0]))]
            (groups[owner] if owner >= 0 else global_points).append(point)
        for index, group in enumerate(groups):
            if len(group) < 3:
                # 区域内证据不足不是判断矛盾：该区域在此波段保持未决，不生成有效测量
                conflicts.append({
                    'kind': 'underconstrained_region',
                    'severity': 'pending',
                    'band': band,
                    'region_id': regions[index]['id'],
                    'confirmed_points': len(group),
                    'message': f"波段{cube.bands[band]['name']}在区域{regions[index]['name']}只有{len(group)}个确认点，该区域此波段将保留未决",
                })
        per_band[band] = {
            'confirmed': len(band_points),
            'pending': len([p for p in points if p['band'] == band and p['status'] == 'pending']),
            'region_points': [len(group) for group in groups],
            'global_points': len(global_points),
        }
    blocking = [item for item in conflicts if item['severity'] == 'blocking']
    return {
        'blocking': len(blocking) > 0,
        'conflicts': conflicts,
        'per_band': per_band,
        'reference_band': reference_band,
        'max_displacement': max_displacement,
        'unconstrained_policy': unconstrained_policy,
        'region_of_pixel': region_of_pixel,
        'regions': regions,
        'points': points,
    }


class FoldError(c.Invalid):
    pass


def _triangulation_group(reference_points, moving_points, label):
    reference = np.asarray(reference_points, dtype=float)
    moving = np.asarray(moving_points, dtype=float)
    if len(reference) < 3:
        raise FoldError(label + '确认点不足3个，无法建立局部变形')
    try:
        triangulation = Delaunay(reference)
    except Exception as error:  # pragma: no cover - scipy raises QhullError
        raise FoldError(label + '对应点共线或过于集中，无法三角剖分') from error
    simplices = triangulation.simplices
    for simplex in simplices:
        ref_orientation = np.cross(
            reference[simplex[1]] - reference[simplex[0]],
            reference[simplex[2]] - reference[simplex[0]],
        )
        mov_orientation = np.cross(
            moving[simplex[1]] - moving[simplex[0]],
            moving[simplex[2]] - moving[simplex[0]],
        )
        if ref_orientation == 0 or mov_orientation == 0 or np.sign(ref_orientation) != np.sign(mov_orientation):
            raise FoldError(label + '存在折叠或翻转三角形，请调整对应点')
    return triangulation, reference, moving


def _barycentric_lookup(points_xy, triangulation, reference, moving):
    """返回每个像素在 moving 点集上的映射坐标；凸包外不映射。

    scipy 的 transform[s, :2] 是 2x2 重心坐标矩阵，transform[s, 2] 是平移项；
    得到的两个权重对应 simplex 的前两个顶点，剩余权重为 1 减两者之和。
    """
    simplex_ids = triangulation.find_simplex(points_xy)
    valid_simplex = np.maximum(simplex_ids, 0)
    delta = points_xy - triangulation.transform[valid_simplex, 2]
    matrices = triangulation.transform[valid_simplex, :2]
    first_weights = np.einsum('ijk,ik->ij', matrices, delta)
    third_weight = 1 - first_weights.sum(axis=1, keepdims=True)
    barycentric = np.concatenate([first_weights, third_weight], axis=1)
    inside = simplex_ids >= 0
    vertices = triangulation.simplices[valid_simplex]
    mapped = np.einsum('ij,ijk->ik', barycentric, moving[vertices])
    mapped[~inside] = points_xy[~inside]
    return mapped, inside, vertices, barycentric, simplex_ids


def build_warp_fields(cube, analysis):
    """按区域独立构建每波段映射；返回 map_x/map_y（输出像素 -> 源图坐标）与状态矩阵。"""
    height, width = cube.shape[:2]
    reference_band = analysis['reference_band']
    regions = analysis['regions']
    points = analysis['points']
    region_of_pixel = analysis['region_of_pixel']
    yy, xx = np.indices((height, width), dtype=float)
    pixel_xy = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1)

    fields = {}
    for band in range(cube.count):
        if band == reference_band:
            continue
        band_points = [p for p in points if p['band'] == band and p['status'] == 'confirmed']
        if not band_points:
            continue
        # 全部几何统一在像素中心坐标系（索引 + 0.5）
        map_x = np.full(height * width, np.nan)
        map_y = np.full(height * width, np.nan)
        supported = np.zeros(height * width, dtype=bool)
        diagnostics = []
        for region_index, region in enumerate(regions):
            group = [p for p in band_points if region_of_pixel[int(round(p['reference'][1])), int(round(p['reference'][0]))] == region_index]
            if len(group) < 3:
                continue
            ref_pts = [[p['reference'][0] + 0.5, p['reference'][1] + 0.5] for p in group]
            mov_pts = [[p['moving'][0] + 0.5, p['moving'][1] + 0.5] for p in group]
            label = f"波段{cube.bands[band]['name']}/区域{region['name']}"
            triangulation, reference, moving = _triangulation_group(ref_pts, mov_pts, label)
            member = (region_of_pixel.ravel() == region_index)
            mapped, inside, vertices, barycentric, simplex_ids = _barycentric_lookup(pixel_xy, triangulation, reference, moving)
            usable = member & inside
            # bilinear_sample 以像素索引（角点）为坐标，映射回索引空间
            map_x[usable] = mapped[usable, 0] - 0.5
            map_y[usable] = mapped[usable, 1] - 0.5
            supported |= usable
            diagnostics.append({'region_id': region['id'], 'points': len(group), 'supported_pixels': int(usable.sum())})
        global_group = [p for p in band_points if region_of_pixel[int(round(p['reference'][1])), int(round(p['reference'][0]))] < 0]
        if len(global_group) >= 3:
            triangulation, reference, moving = _triangulation_group(
                [[p['reference'][0] + 0.5, p['reference'][1] + 0.5] for p in global_group],
                [[p['moving'][0] + 0.5, p['moving'][1] + 0.5] for p in global_group],
                f"波段{cube.bands[band]['name']}/全局",
            )
            mapped, inside, *_ = _barycentric_lookup(pixel_xy, triangulation, reference, moving)
            free = ~supported & inside
            map_x[free] = mapped[free, 0] - 0.5
            map_y[free] = mapped[free, 1] - 0.5
            supported |= free
            diagnostics.append({'region_id': None, 'points': len(global_group), 'supported_pixels': int(free.sum())})
        fields[band] = {
            'map_x': map_x.reshape(height, width),
            'map_y': map_y.reshape(height, width),
            'supported': supported.reshape(height, width),
            'diagnostics': diagnostics,
        }
    return fields


def apply_local_warp(source, analysis, fields):
    height, width = source.shape[:2]
    result = np.zeros_like(source.values)
    output_valid = np.zeros_like(source.valid)
    reference_band = analysis['reference_band']
    policy = analysis['unconstrained_policy']
    yy, xx = np.indices((height, width), dtype=float)
    reports = []
    for band in range(source.count):
        if band == reference_band:
            result[:, :, band] = source.values[:, :, band]
            output_valid[:, :, band] = source.valid[:, :, band]
            reports.append({'band': source.bands[band]['name'], 'status': 'reference', 'warped_pixels': 0,
                            'unconstrained_pixels': 0, 'pending_pixels': 0, 'valid_count': int(source.valid[:, :, band].sum())})
            continue
        field = fields.get(band)
        confirmed = [p for p in analysis['points'] if p['band'] == band and p['status'] == 'confirmed']
        if field is None:
            values, valid, warped, unconstrained, pending = _passthrough_band(source, band, policy)
            status = 'pending_passthrough' if policy == 'passthrough' else 'pending_masked'
            reports.append(_band_report(source, band, values, valid, status, 0, unconstrained, pending,
                                        reason='没有任何人工确认对应点，位置无法可靠确定'))
            result[:, :, band] = values
            output_valid[:, :, band] = valid
            continue
        map_x = field['map_x']
        map_y = field['map_y']
        supported = field['supported']
        # 凸包外映射为 NaN；这些像素由 supported 门控，填零仅为避免插值转换告警
        safe_x = np.where(supported, map_x, 0.0)
        safe_y = np.where(supported, map_y, 0.0)
        sampled_values, sampled_valid = bilinear_sample(
            source.values[:, :, band], source.valid[:, :, band], safe_x, safe_y, strict=True,
        )
        warped = supported & sampled_valid
        values = np.zeros((height, width), dtype=np.float32)
        valid = np.zeros((height, width), dtype=bool)
        values[warped] = sampled_values[warped].astype(np.float32)
        valid[warped] = True
        free = ~supported
        if policy == 'passthrough':
            values[free] = source.values[free, band]
            valid[free] = source.valid[free, band]
            unconstrained = int((free & source.valid[:, :, band]).sum())
            pending = 0
        else:
            unconstrained = 0
            pending = int(free.sum())
        # 严格支持失败（源邻域缺测/越界）也属于无法可靠确定，按未决处理
        failed = supported & ~sampled_valid
        pending += int(failed.sum())
        displacement = np.sqrt((map_x - xx) ** 2 + (map_y - yy) ** 2)
        common_before = source.valid[:, :, reference_band] & source.valid[:, :, band]
        common_after = valid & source.valid[:, :, reference_band]
        reference = source.values[:, :, reference_band]
        reports.append(_band_report(
            source, band, values, valid, 'warped', int(warped.sum()), unconstrained, pending,
            displacement=float(np.nanmax(np.where(supported, displacement, np.nan))) if supported.any() else 0.0,
            mean_displacement=float(np.nanmean(np.where(supported, displacement, np.nan))) if supported.any() else 0.0,
            before=float(np.mean(np.abs(source.values[:, :, band][common_before] - reference[common_before]))) if common_before.any() else None,
            after=float(np.mean(np.abs(values[common_after] - reference[common_after]))) if common_after.any() else None,
            diagnostics=field['diagnostics'], confirmed_points=len(confirmed),
        ))
        result[:, :, band] = values
        output_valid[:, :, band] = valid
    return result, output_valid, reports


def _passthrough_band(source, band, policy):
    height, width = source.shape[:2]
    if policy == 'passthrough':
        values = source.values[:, :, band].copy()
        valid = source.valid[:, :, band].copy()
        return values, valid, 0, int(valid.sum()), 0
    return np.zeros((height, width), np.float32), np.zeros((height, width), bool), 0, 0, height * width


def _band_report(source, band, values, valid, status, warped, unconstrained, pending, reason=None,
                 displacement=None, mean_displacement=None, before=None, after=None, diagnostics=None, confirmed_points=0):
    row = {
        'band': source.bands[band]['name'],
        'band_index': band,
        'status': status,
        'confirmed_points': confirmed_points,
        'warped_pixels': warped,
        'unconstrained_pixels': unconstrained,
        'pending_pixels': pending,
        'valid_count': int(valid.sum()),
        'reason': reason,
    }
    if displacement is not None:
        row['max_displacement_px'] = round(displacement, 4)
        row['mean_displacement_px'] = round(mean_displacement, 4)
    if before is not None:
        row['before_absolute_difference'] = before
        row['after_absolute_difference'] = after
    if diagnostics is not None:
        row['warp_groups'] = diagnostics
    return row


def normalized_cross_correlation(reference, moving, valid_ref, valid_mov, center_x, center_y,
                                 template_radius, search_radius):
    height, width = reference.shape
    x0 = int(round(center_x)) - template_radius
    y0 = int(round(center_y)) - template_radius
    x1 = x0 + 2 * template_radius + 1
    y1 = y0 + 2 * template_radius + 1
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise c.Invalid('模板窗口超出图像边界，请把参考点移向图像内部或缩小模板')
    template = reference[y0:y1, x0:x1].astype(np.float64)
    template_valid = valid_ref[y0:y1, x0:x1]
    if int(template_valid.sum()) < MIN_TEMPLATE_VALID:
        raise c.Invalid('参考点周围有效像素不足，纹理可能在此波段消失')
    mean = template[template_valid].mean()
    centered = np.where(template_valid, template - mean, 0.0)
    norm = np.sqrt((centered[template_valid] ** 2).sum())
    if norm < 1e-10:
        raise c.Invalid('参考点周围纹理无变化（该波长下纹理可能已消失），无法可靠匹配')
    candidates = []
    for offset_y in range(-search_radius, search_radius + 1):
        for offset_x in range(-search_radius, search_radius + 1):
            mx0, my0 = x0 + offset_x, y0 + offset_y
            mx1, my1 = x1 + offset_x, y1 + offset_y
            if mx0 < 0 or my0 < 0 or mx1 > width or my1 > height:
                continue
            patch = moving[my0:my1, mx0:mx1].astype(np.float64)
            patch_valid = valid_mov[my0:my1, mx0:mx1] & template_valid
            if int(patch_valid.sum()) < MIN_TEMPLATE_VALID:
                continue
            p_mean = patch[patch_valid].mean()
            p_centered = np.where(patch_valid, patch - p_mean, 0.0)
            numerator = (centered * p_centered)[patch_valid].sum()
            denominator = norm * np.sqrt((p_centered[patch_valid] ** 2).sum())
            if denominator < 1e-10:
                continue
            score = float(numerator / denominator)
            candidates.append({
                'moving': [float(mx0 + template_radius), float(my0 + template_radius)],
                'score': round(score, 5),
                'ambiguous': False,
                'reason': '归一化互相关',
            })
    if not candidates:
        raise c.Invalid('搜索范围内没有满足有效覆盖的候选位置')
    candidates.sort(key=lambda item: item['score'], reverse=True)
    return candidates[:SESSION_LIMIT_CANDIDATES]


def suggest_candidates(cube, reference_band, band, reference_xy, template_radius, search_radius, ambiguity_margin):
    height, width = cube.shape[:2]
    if band == reference_band:
        raise c.Invalid('参考波段不需要候选')
    template_radius = c.integer(template_radius, '模板半径', MIN_TEMPLATE_SIDE // 2, MAX_TEMPLATE_SIDE // 2)
    search_radius = c.integer(search_radius, '搜索半径', 1, min(100, min(width, height) // 3))
    ambiguity_margin = c.number(ambiguity_margin, '歧义分差阈值', 0.0, 1.0)
    reference = cube.values[:, :, reference_band].astype(float)
    moving = cube.values[:, :, band].astype(float)
    valid_ref = cube.valid[:, :, reference_band]
    valid_mov = cube.valid[:, :, band]
    candidates = normalized_cross_correlation(
        reference, moving, valid_ref, valid_mov,
        reference_xy[0], reference_xy[1], template_radius, search_radius,
    )
    top = candidates[0]['score']
    for index, item in enumerate(candidates):
        item['rank'] = index
        item['ambiguous'] = bool(index > 0 and (top - item['score']) <= ambiguity_margin) or (
            index == 0 and len(candidates) > 1 and (candidates[0]['score'] - candidates[1]['score']) <= ambiguity_margin
        )
    if candidates[0]['ambiguous']:
        candidates[0]['reason'] = '归一化互相关；与次高分差过小，位置存在歧义'
    else:
        candidates[0]['reason'] = '归一化互相关；与次高分差超过歧义阈值'
    return candidates


def translation_hint(cube, reference_band, band, max_shift):
    common = cube.valid[:, :, reference_band] & cube.valid[:, :, band]
    try:
        dx, dy, score = phase_translation(
            cube.values[:, :, reference_band].astype(float),
            cube.values[:, :, band].astype(float),
            common, max_shift,
        )
    except c.Invalid as error:
        return {'available': False, 'reason': str(error)}
    return {'available': True, 'dx': dx, 'dy': dy, 'score': float(score),
            'meaning': '仅作审阅提示，不写入对应点，也不约束配准结果'}
