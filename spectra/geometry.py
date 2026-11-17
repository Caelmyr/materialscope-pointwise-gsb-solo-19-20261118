from collections import defaultdict

import numpy as np

from . import contracts as c


def pixel_edges(mask):
    edges = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        x = int(x)
        y = int(y)
        if y == 0 or not mask[y - 1, x]:
            edges.append(((x, y), (x + 1, y)))
        if x == width - 1 or not mask[y, x + 1]:
            edges.append(((x + 1, y), (x + 1, y + 1)))
        if y == height - 1 or not mask[y + 1, x]:
            edges.append(((x + 1, y + 1), (x, y + 1)))
        if x == 0 or not mask[y, x - 1]:
            edges.append(((x, y + 1), (x, y)))
    return edges


def signed_area(ring):
    values = np.asarray(ring, dtype=float)
    return float(np.sum(values[:-1, 0] * values[1:, 1] - values[1:, 0] * values[:-1, 1]) / 2)


def simplify_collinear(ring):
    points = ring[:-1]
    if len(points) <= 3:
        return ring
    output = []
    for index, current in enumerate(points):
        previous = points[index - 1]
        following = points[(index + 1) % len(points)]
        cross = (current[0] - previous[0]) * (following[1] - current[1]) - (current[1] - previous[1]) * (following[0] - current[0])
        if cross != 0:
            output.append(current)
    if not output:
        return ring
    return output + [output[0]]


def boundary_rings(mask):
    edges = pixel_edges(mask)
    outgoing = defaultdict(list)
    for start, end in edges:
        outgoing[start].append(end)
    remaining = set(edges)
    rings = []
    directions = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    while remaining:
        first = min(remaining)
        start, current = first
        remaining.remove(first)
        ring = [start, current]
        previous = start
        while current != start:
            candidates = [end for end in outgoing[current] if (current, end) in remaining]
            if not candidates:
                raise c.Invalid('对象边界不闭合')
            incoming = directions[(current[0] - previous[0], current[1] - previous[1])]
            def priority(end):
                direction = directions[(end[0] - current[0], end[1] - current[1])]
                turn = (direction - incoming) % 4
                return {1: 0, 0: 1, 3: 2, 2: 3}[turn]
            following = min(candidates, key=priority)
            remaining.remove((current, following))
            previous, current = current, following
            ring.append(current)
            if len(ring) > len(edges) + 1:
                raise c.Invalid('对象边界追踪超出边数')
        rings.append(simplify_collinear(ring))
    return rings


def point_in_ring(point, ring):
    x, y = point
    inside = False
    for first, second in zip(ring[:-1], ring[1:]):
        x1, y1 = first
        x2, y2 = second
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
    return inside


def group_rings(rings):
    outer = [ring for ring in rings if signed_area(ring) > 0]
    holes = [ring for ring in rings if signed_area(ring) < 0]
    polygons = [[ring] for ring in outer]
    for hole in holes:
        first, second = hole[0], hole[1]
        midpoint = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
        candidates = [index for index, ring in enumerate(outer) if point_in_ring(midpoint, ring)]
        if not candidates:
            raise c.Invalid('孔洞边界缺少所属外环')
        chosen = min(candidates, key=lambda index: abs(signed_area(outer[index])))
        polygons[chosen].append(hole)
    return polygons


def label_geometry(cube, options):
    if cube.domain != 'labels' or cube.count != 1:
        raise c.Invalid('对象轮廓导出需要单波段标签图')
    field = cube.values[:, :, 0]
    valid = cube.valid[:, :, 0]
    if np.any(field[valid] < 0) or np.any(field[valid] != np.floor(field[valid])):
        raise c.Invalid('标签必须是非负整数')
    labels = np.unique(field[valid].astype(int))
    labels = labels[labels != 0]
    if len(labels) > 1000:
        raise c.Invalid('轮廓导出最多1000个标签')
    selected = options.get('labels')
    if selected is not None:
        selected = [c.integer(item, '标签', 1, 1000000) for item in c.array_value(selected, '标签', 1, 1000)]
        labels = labels[np.isin(labels, selected)]
    features = []
    edge_budget = 200000
    used_edges = 0
    for label in labels:
        mask = (field == label) & valid
        rings = boundary_rings(mask)
        used_edges += sum(len(ring) for ring in rings)
        if used_edges > edge_budget:
            raise c.Invalid('对象轮廓超过20万顶点预算，请筛选对象后导出')
        polygons = group_rings(rings)
        transformed = []
        for polygon in polygons:
            transformed.append([
                [[cube.grid['origin'][0] + x * cube.grid['x_step'], cube.grid['origin'][1] + y * cube.grid['y_step']] for x, y in ring]
                for ring in polygon
            ])
        features.append({
            'type': 'Feature',
            'properties': {'label': int(label), 'pixels': int(mask.sum()), 'area': float(mask.sum() * cube.grid['x_step'] * cube.grid['y_step']), 'unit': cube.grid['unit']},
            'geometry': {'type': 'MultiPolygon', 'coordinates': transformed},
        })
    return {
        'type': 'FeatureCollection',
        'coordinate_system': 'local image coordinates, x right, y down; not RFC7946 geographic coordinates',
        'boundary_semantics': 'exact exposed pixel edges, collinear vertices simplified, holes retained',
        'features': features,
        'source_fingerprint': cube.fingerprint(),
    }
