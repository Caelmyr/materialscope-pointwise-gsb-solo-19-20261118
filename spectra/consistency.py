import copy
import json

import numpy as np

from . import contracts as c


def bounded_json(value, label, maximum_bytes):
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise c.Invalid(label + '含非有限或非JSON数据') from None
    if len(encoded.encode()) > maximum_bytes:
        raise c.Invalid(label + '超过容量')
    return value


def unique_rows(value, label, maximum):
    rows = c.array_value(value, label, 0, maximum)
    seen = set()
    for row in rows:
        c.object_value(row, label + '条目')
        identifier = c.identifier(row.get('id'), label + '标识')
        if identifier in seen:
            raise c.Invalid(label + '标识重复')
        seen.add(identifier)
    return rows


def validate_asset(row):
    c.identifier(row.get('id'), '影像标识')
    c.text(row.get('name'), '影像名称')
    c.text(row.get('created_at'), '创建时间', 64)
    digest = c.text(row.get('digest'), '素材摘要', 64)
    if len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
        raise c.Invalid('影像摘要不合法')
    parent = row.get('parent_id')
    if parent is not None:
        c.identifier(parent, '父影像')
    c.text(row.get('operation'), '来源操作', 80)
    c.options_copy(row.get('parameters', {}))
    bounded_json(row.get('report', {}), '处理报告', 5000000)
    summary = c.object_value(row.get('summary'), '影像摘要')
    width = c.integer(summary.get('width'), '宽度', 1, c.LIMITS['width'])
    height = c.integer(summary.get('height'), '高度', 1, c.LIMITS['height'])
    bands = c.integer(summary.get('band_count'), '波段数', 1, c.LIMITS['bands'])
    if width * height * bands > c.LIMITS['samples']:
        raise c.Invalid('影像摘要样本数超界')
    c.band_metadata(summary.get('bands'), bands)
    c.grid_metadata(summary.get('grid'))
    c.choice(summary.get('domain'), '数值域', ['raw', 'reflectance', 'derived', 'labels'])
    c.text(summary.get('unit'), '数值单位', 32)
    for key in ('common_valid_count', 'any_valid_count'):
        c.integer(summary.get(key), key, 0, width * height)
    counts = c.array_value(summary.get('valid_counts'), '逐带有效计数', bands, bands)
    for count in counts:
        c.integer(count, '有效计数', 0, width * height)
    if summary['common_valid_count'] > summary['any_valid_count']:
        raise c.Invalid('共同有效计数不能超过任意有效计数')
    return row


def validate_parent_graph(assets):
    lookup = {item['id']: item for item in assets}
    for asset in assets:
        current = asset['id']
        visited = set()
        while current is not None:
            if current in visited:
                raise c.Invalid('影像父来源存在循环')
            if current not in lookup:
                raise c.Invalid('影像父来源不在本状态内')
            visited.add(current)
            current = lookup[current].get('parent_id')
    return lookup


def validate_region(row, assets):
    c.identifier(row.get('id'), '分区标识')
    asset_id = c.identifier(row.get('asset_id'), '分区影像')
    if asset_id not in assets:
        raise c.Invalid('分区引用不存在的影像')
    summary = assets[asset_id]['summary']
    c.text(row.get('name'), '分区名称')
    c.choice(row.get('kind'), '分区形状', ['rectangle', 'ellipse'])
    c.bounds(row.get('bounds'), summary['width'], summary['height'])
    c.color(row.get('color'))
    c.text(row.get('created_at'), '分区创建时间', 64)
    training = row.get('training')
    if training is not None:
        training = c.object_value(training, '训练标记')
        role = c.choice(training.get('role'), '标记角色', ['class', 'negative'])
        if role == 'class':
            c.text(training.get('name'), '材料名称', 60)
    return row


def validate_classifier(row):
    c.identifier(row.get('id'), '分类器标识')
    c.text(row.get('name'), '分类器名称')
    c.text(row.get('created_at'), '分类器创建时间', 64)
    c.identifier(row.get('source_asset_id'), '分类器来源影像')
    fingerprint = c.text(row.get('source_fingerprint'), '分类器来源摘要', 64)
    if len(fingerprint) != 64 or any(char not in '0123456789abcdef' for char in fingerprint):
        raise c.Invalid('分类器来源摘要不合法')
    c.choice(row.get('orientation'), '误判取向', ['pending', 'foreign_sensitive', 'known_conservative'])
    waves = c.finite_array(row.get('wavelengths_nm'), '分类器波长', 1, 64)
    if len(waves) < 3 or np.any(np.diff(waves) <= 0):
        raise c.Invalid('分类器波长需至少3个且严格递增')
    classes = c.array_value(row.get('classes'), '材料类', 1, 32)
    names = set()
    for item in classes:
        c.identifier(item.get('id'), '材料类标识')
        name = c.text(item.get('name'), '材料类名称', 60)
        if name in names:
            raise c.Invalid('材料类名称重复')
        names.add(name)
        c.integer(item.get('pixels'), '材料类像素', 0, c.LIMITS['samples'])
        c.array_value(item.get('region_ids'), '材料类区域', 1, c.LIMITS['regions'])
    negative = c.object_value(row.get('negative'), '反例统计')
    c.integer(negative.get('pixels'), '反例像素', 0, c.LIMITS['samples'])
    model = c.object_value(row.get('model'), '分类模型')
    centers = c.finite_array(model.get('centers'), '类中心', 2, 100000)
    if centers.shape != (len(classes), len(waves)):
        raise c.Invalid('类中心形状与类别、波长不一致')
    covariance = c.finite_array(model.get('covariance'), '类内协方差', 2, 100000)
    if covariance.shape != (len(waves), len(waves)):
        raise c.Invalid('类内协方差形状与波长不一致')
    key_bands = c.array_value(model.get('key_bands'), '关键波段', 0, len(waves))
    for band in key_bands:
        c.integer(band, '关键波段', 0, len(waves) - 1)
    c.integer(model.get('min_bands'), '最少有效波段', 1, len(waves))
    regions = c.array_value(row.get('training_regions'), '训练区域快照', 1, c.LIMITS['regions'])
    for item in regions:
        c.identifier(item.get('id'), '训练区域标识')
        c.text(item.get('name'), '训练区域名称')
        c.choice(item.get('kind'), '训练区域形状', ['rectangle', 'ellipse'])
        c.array_value(item.get('bounds'), '训练区域边界', 4, 4)
        c.choice(item.get('role'), '训练区域角色', ['class', 'negative'])
    validation = c.object_value(row.get('validation'), '分类器验证')
    c.boolean(validation.get('sufficient'), '验证充分性')
    bounded_json(row, '分类器', 1500000)
    return row


def validate_signature(row):
    c.identifier(row.get('id'), '材料参考标识')
    c.text(row.get('name'), '材料名称')
    wavelengths = c.finite_array(row.get('wavelengths_nm'), '参考波长', 1, 512)
    values = c.finite_array(row.get('values'), '参考光谱', 1, 512)
    if len(wavelengths) < 3 or len(wavelengths) != len(values):
        raise c.Invalid('参考光谱需至少3个对应采样点')
    if np.any(np.diff(wavelengths) <= 0):
        raise c.Invalid('保存的参考波长必须严格递增')
    if np.any(wavelengths < 100) or np.any(wavelengths > 30000):
        raise c.Invalid('参考波长超界')
    if row.get('domain') != 'reflectance' or row.get('unit') != 'fraction':
        raise c.Invalid('材料参考须为fraction反射率')
    for key in ('std', 'source_counts'):
        if row.get(key) is not None:
            vector = c.finite_array(row[key], key, 1, 512)
            if len(vector) != len(values) or np.any(vector < 0):
                raise c.Invalid('参考附加统计长度或范围无效')
    source_region = row.get('source_region')
    if source_region is not None:
        c.identifier(source_region, '参考来源分区')
    c.text(row.get('created_at'), '参考创建时间', 64)
    return row


def validate_state(state):
    state = c.object_value(state, '工程状态')
    c.text(state.get('name'), '工程名称')
    c.text(state.get('notes'), '工程说明', 10000, True)
    assets = unique_rows(state.get('assets'), '影像', c.LIMITS['assets'])
    for row in assets:
        validate_asset(row)
    lookup = validate_parent_graph(assets)
    region_rows = unique_rows(state.get('regions'), '分区', c.LIMITS['regions'])
    for row in region_rows:
        validate_region(row, lookup)
    signatures = unique_rows(state.get('signatures'), '材料参考', c.LIMITS['signatures'])
    for row in signatures:
        validate_signature(row)
    classifiers = unique_rows(state.get('classifiers') or [], '分类器', c.LIMITS['classifiers'])
    for row in classifiers:
        validate_classifier(row)
    return state


def validate_project_structure(project):
    project = c.object_value(project, '工程')
    if project.get('schema') != 'materialscope_project_v1':
        raise c.Invalid('工程版本不支持')
    c.identifier(project.get('id'), '工程标识')
    c.revision(project.get('revision'))
    c.text(project.get('created_at'), '创建时间', 64)
    c.text(project.get('updated_at'), '更新时间', 64)
    validate_state(project)
    snapshots = unique_rows(project.get('snapshots'), '命名版本', 20)
    for snapshot in snapshots:
        c.text(snapshot.get('name'), '版本名称')
        c.text(snapshot.get('created_at'), '版本时间', 64)
        validate_state(snapshot.get('state'))
    for key in ('history', 'redo'):
        history = c.array_value(project.get(key), key, 0, c.LIMITS['history'])
        for entry in history:
            c.text(entry.get('action'), '历史操作', 200)
            c.text(entry.get('time'), '历史时间', 64)
            validate_state(entry.get('state'))
    return project


def compare_states(before, after):
    before = validate_state(before)
    after = validate_state(after)
    result = {'name_changed': before['name'] != after['name'], 'notes_changed': before['notes'] != after['notes']}
    for key in ('assets', 'regions', 'signatures', 'classifiers'):
        left = {row['id']: row for row in before.get(key) or []}
        right = {row['id']: row for row in after.get(key) or []}
        common = set(left) & set(right)
        result[key] = {
            'added': sorted(set(right) - set(left)),
            'removed': sorted(set(left) - set(right)),
            'changed': sorted(identifier for identifier in common if left[identifier] != right[identifier]),
            'unchanged': sorted(identifier for identifier in common if left[identifier] == right[identifier]),
        }
    return result


def compare_snapshot(project, snapshot_id):
    validate_project_structure(project)
    snapshot = next((item for item in project['snapshots'] if item['id'] == snapshot_id), None)
    if snapshot is None:
        raise c.Invalid('命名版本不存在')
    return {'snapshot_id': snapshot_id, 'snapshot_name': snapshot['name'], 'current_revision': project['revision'], 'changes_since_snapshot': compare_states(snapshot['state'], project)}


def project_dependencies(project):
    validate_project_structure(project)
    assets = {item['id']: item for item in project['assets']}
    children = {identifier: [] for identifier in assets}
    regions = {identifier: [] for identifier in assets}
    for asset in assets.values():
        if asset.get('parent_id'):
            children[asset['parent_id']].append(asset['id'])
    for region in project['regions']:
        regions[region['asset_id']].append(region['id'])
    output = []
    for identifier, asset in assets.items():
        output.append({
            'asset_id': identifier,
            'name': asset['name'],
            'parent_id': asset.get('parent_id'),
            'child_ids': children[identifier],
            'region_ids': regions[identifier],
            'can_remove_leaf': not children[identifier],
            'snapshot_references': [snapshot['id'] for snapshot in project['snapshots'] if any(row['id'] == identifier for row in snapshot['state']['assets'])],
        })
    return {
        'assets': output,
        'classifiers': [
            {
                'id': row['id'],
                'name': row['name'],
                'source_asset_id': row['source_asset_id'],
                'orientation': row['orientation'],
                'class_count': len(row['classes']),
            }
            for row in project.get('classifiers', [])
        ],
        'material_reference_policy': 'references are immutable measured spectra; removing a region does not delete its saved measurement',
        'classifier_policy': 'classifiers freeze training regions, labels, wavelengths, validation split and chosen orientation; later region or library edits never alter saved classifiers or their results',
    }
