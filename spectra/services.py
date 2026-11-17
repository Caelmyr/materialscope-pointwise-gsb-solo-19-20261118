import copy
import hashlib
import json
import time

import numpy as np

from . import contracts as c
from . import diagnostics, library, regions
from .demo import create_demo
from .engine import run_operation, OPERATIONS
from .io import import_cube, read_signature_csv
from .model import metadata_replacement
from .rendering import default_view


class Services:
    def __init__(self, store):
        self.store = store

    def project_summary(self, project_id):
        project = self.store.get(project_id)
        output = copy.deepcopy(project)
        output['undo_count'] = len(project['history'])
        output['redo_count'] = len(project['redo'])
        output['history'] = [{'action': item['action'], 'time': item['time']} for item in project['history']]
        output['redo'] = [{'action': item['action'], 'time': item['time']} for item in project['redo']]
        output['snapshots'] = [{'id': item['id'], 'name': item['name'], 'created_at': item['created_at'], 'asset_count': len(item['state']['assets'])} for item in project['snapshots']]
        return output

    def asset_detail(self, project_id, asset_id):
        project, asset, cube = self.store.asset(project_id, asset_id)
        return {
            'asset': asset,
            'revision': project['revision'],
            'view': default_view(cube),
            'regions': [item for item in project['regions'] if item['asset_id'] == asset_id],
            'capabilities': diagnostics.processing_capabilities(cube),
            'metadata_audit': diagnostics.metadata_audit(cube),
        }

    def import_asset(self, project_id, expected_revision, data, filename, name=None):
        cube = import_cube(data, filename)
        name = name or filename
        project, row = self.store.append_cube(project_id, expected_revision, cube, name, operation='import', report={
            'filename': filename,
            'uploaded_bytes': len(data),
            'source_byte_sha256': hashlib.sha256(data).hexdigest(),
            'raw_upload_retained': False,
        })
        return {'project': self.project_summary(project_id), 'asset': row}

    def load_demo(self, name='材料分析示范'):
        project = self.store.create(name)
        cubes, signatures, region_specs = create_demo()
        names = {
            'raw': '01 · 原始材料试片',
            'dark': '02 · 暗场参考',
            'white': '03 · 白板参考',
            'reflectance': '04 · 合成反射率试片',
            'misregistered': '05 · 波段错位试片',
        }
        assets = {}
        for role, cube in cubes.items():
            project, row = self.store.append_cube(project['id'], project['revision'], cube, names[role], operation='synthetic_demo', report={'role': role, 'synthetic': True})
            assets[role] = row
        with self.store.transaction(project['id'], project['revision'], '装载示范分区与材料参考') as current:
            current['signatures'].extend(signatures)
            for spec in region_specs:
                row = regions.validate_region(cubes['reflectance'], assets['reflectance']['id'], spec)
                current['regions'].append(row)
            current['notes'] = '程序合成材料试片，用于本地功能演示；不是现实观测，也不是实测材料谱库。'
        return {'project': self.project_summary(project['id']), 'roles': {key: item['id'] for key, item in assets.items()}}

    def _references(self, project, source_asset, operation, options):
        references = {}
        assets = {item['id']: item for item in project['assets']}
        for role in ('dark', 'white'):
            key = role + '_id'
            if options.get(key):
                identifier = c.identifier(options[key], role + '参考')
                if identifier not in assets:
                    raise c.Invalid('参考影像不在当前工程')
                references[role] = self.store.load_cube(assets[identifier]['digest'])
        if operation == 'match_library':
            selected = options.get('signature_ids')
            if selected is None:
                references['signatures'] = project['signatures']
            else:
                selected = c.array_value(selected, '材料参考', 1, 100)
                lookup = {row['id']: row for row in project['signatures']}
                if any(identifier not in lookup for identifier in selected):
                    raise c.Invalid('材料参考不存在')
                references['signatures'] = [lookup[identifier] for identifier in selected]
        return references

    def execute(self, project_id, asset_id, expected_revision, operation, options, name=None):
        project, asset, cube = self.store.asset(project_id, asset_id)
        if project['revision'] != c.revision(expected_revision):
            raise c.Conflict('工程已变化，请刷新后执行')
        references = self._references(project, asset, operation, options)
        started = time.monotonic()
        output, report, parameters = run_operation(cube, operation, options, references)
        elapsed = time.monotonic() - started
        report['elapsed_seconds'] = elapsed
        reference_digests = {}
        for role in ('dark', 'white'):
            if role in references:
                reference_digests[role] = references[role].fingerprint()
        report['reference_fingerprints'] = reference_digests
        if references.get('signatures'):
            report['signature_snapshot'] = copy.deepcopy(references['signatures'])
        title = c.text(name or (asset['name'] + ' · ' + OPERATIONS[operation][0]), '结果名称')
        updated, row = self.store.append_cube(project_id, expected_revision, output, title, asset_id, operation, parameters, report)
        return {'project': self.project_summary(project_id), 'asset': row}

    def modify_metadata(self, project_id, asset_id, expected_revision, options):
        project, asset, cube = self.store.asset(project_id, asset_id)
        updated = metadata_replacement(cube, options)
        _, row = self.store.append_cube(project_id, expected_revision, updated, asset['name'] + ' · 元数据修订', asset_id, 'metadata', options, {'previous_metadata': cube.summary()})
        return {'project': self.project_summary(project_id), 'asset': row}

    def create_region(self, project_id, asset_id, expected_revision, options):
        project, asset, cube = self.store.asset(project_id, asset_id)
        row = regions.validate_region(cube, asset_id, options)
        with self.store.transaction(project_id, expected_revision, '新增取样分区') as current:
            current['regions'].append(row)
        return {'project': self.project_summary(project_id), 'region': row, 'report': regions.region_report(cube, row)}

    def update_region(self, project_id, region_id, expected_revision, options):
        project = self.store.get(project_id)
        original = next((row for row in project['regions'] if row['id'] == region_id), None)
        if original is None:
            raise c.Invalid('分区不存在')
        _, asset, cube = self.store.asset(project_id, original['asset_id'])
        merged = {**original, **options}
        replacement = regions.validate_region(cube, original['asset_id'], merged)
        replacement['id'] = original['id']
        replacement['created_at'] = original['created_at']
        with self.store.transaction(project_id, expected_revision, '修订取样分区') as current:
            current['regions'] = [replacement if row['id'] == region_id else row for row in current['regions']]
        return {'project': self.project_summary(project_id), 'region': replacement, 'report': regions.region_report(cube, replacement)}

    def delete_region(self, project_id, region_id, expected_revision):
        with self.store.transaction(project_id, expected_revision, '移除取样分区') as project:
            if not any(row['id'] == region_id for row in project['regions']):
                raise c.Invalid('分区不存在')
            project['regions'] = [row for row in project['regions'] if row['id'] != region_id]
        return self.project_summary(project_id)

    def region(self, project_id, region_id):
        project = self.store.get(project_id)
        region = next((row for row in project['regions'] if row['id'] == region_id), None)
        if region is None:
            raise c.Invalid('分区不存在')
        _, _, cube = self.store.asset(project_id, region['asset_id'])
        return region, cube

    def region_report(self, project_id, region_id):
        region, cube = self.region(project_id, region_id)
        return regions.region_report(cube, region)

    def save_region_signature(self, project_id, region_id, expected_revision, name):
        region, cube = self.region(project_id, region_id)
        signature = library.signature_from_region(cube, region, name)
        with self.store.transaction(project_id, expected_revision, '从分区保存材料参考') as project:
            project['signatures'].append(signature)
        return {'project': self.project_summary(project_id), 'signature': signature}

    def import_signature(self, project_id, expected_revision, options):
        signature = library.validate_signature(options)
        with self.store.transaction(project_id, expected_revision, '导入材料参考') as project:
            project['signatures'].append(signature)
        return {'project': self.project_summary(project_id), 'signature': signature}

    def import_signature_csv(self, project_id, expected_revision, data, name):
        values = read_signature_csv(data)
        return self.import_signature(project_id, expected_revision, {**values, 'name': name})

    def delete_signature(self, project_id, expected_revision, signature_id):
        with self.store.transaction(project_id, expected_revision, '移除材料参考') as project:
            if not any(row['id'] == signature_id for row in project['signatures']):
                raise c.Invalid('材料参考不存在')
            project['signatures'] = [row for row in project['signatures'] if row['id'] != signature_id]
        return self.project_summary(project_id)

    def compare_signatures(self, project_id, selected):
        project = self.store.get(project_id)
        selected = c.array_value(selected, '材料参考', 2, 20)
        lookup = {row['id']: row for row in project['signatures']}
        if any(item not in lookup for item in selected):
            raise c.Invalid('材料参考不存在')
        return library.signature_comparison([lookup[item] for item in selected])

    def lineage(self, project_id, asset_id):
        project = self.store.get(project_id)
        lookup = {row['id']: row for row in project['assets']}
        if asset_id not in lookup:
            raise c.Invalid('影像不存在')
        current = asset_id
        visited = set()
        rows = []
        while current:
            if current in visited:
                raise c.Invalid('影像来源链存在循环')
            visited.add(current)
            row = lookup[current]
            rows.append({
                'id': row['id'],
                'name': row['name'],
                'digest': row['digest'],
                'operation': row['operation'],
                'parameters': row['parameters'],
                'created_at': row['created_at'],
                'domain': row['summary']['domain'],
                'unit': row['summary']['unit'],
            })
            current = row.get('parent_id')
            if current and current not in lookup:
                raise c.Invalid('影像来源链中断')
        return {'asset_id': asset_id, 'chain': rows, 'project_revision': project['revision']}

    def compare_assets(self, project_id, first_id, second_id, options):
        _, first_asset, first = self.store.asset(project_id, first_id)
        _, second_asset, second = self.store.asset(project_id, second_id)
        if first.shape != second.shape or first.grid != second.grid:
            raise c.Invalid('数值比较要求形状与空间网格相同')
        if first.domain != second.domain or first.unit != second.unit:
            raise c.Invalid('数值比较要求数值域与单位一致')
        if list(first.bands) != list(second.bands):
            raise c.Invalid('数值比较要求波段定义一致')
        reports = []
        for band in range(first.count):
            valid = first.valid[:, :, band] & second.valid[:, :, band]
            if not valid.any():
                reports.append({'band': first.bands[band]['name'], 'common_count': 0})
                continue
            a = first.values[:, :, band][valid].astype(float)
            b = second.values[:, :, band][valid].astype(float)
            delta = b - a
            reports.append({
                'band': first.bands[band]['name'],
                'common_count': int(valid.sum()),
                'first_only_count': int((first.valid[:, :, band] & ~second.valid[:, :, band]).sum()),
                'second_only_count': int((second.valid[:, :, band] & ~first.valid[:, :, band]).sum()),
                'bias': float(delta.mean()),
                'mean_absolute_difference': float(np.abs(delta).mean()),
                'root_mean_square_difference': float(np.sqrt(np.mean(delta ** 2))),
                'maximum_absolute_difference': float(np.abs(delta).max()),
            })
        return {'first': first_asset['name'], 'second': second_asset['name'], 'unit': first.unit, 'bands': reports}

    def execute_recipe(self, project_id, asset_id, expected_revision, steps, name):
        steps = c.array_value(steps, '配方步骤', 1, 6)
        project, asset, cube = self.store.asset(project_id, asset_id)
        if project['revision'] != expected_revision:
            raise c.Conflict('工程已变化')
        prepared = []
        current = cube
        parent = asset_id
        for position, step in enumerate(steps):
            step = c.object_value(step, '配方步骤')
            operation = c.text(step.get('operation'), '操作')
            options = c.options_copy(step.get('options', {}))
            refs = self._references(project, asset, operation, options)
            output, report, parameters = run_operation(current, operation, options, refs)
            digest = self.store.save_cube(output)
            row = {
                'id': c.generated_id('cube'),
                'name': c.text(f'{name} · {position + 1} {OPERATIONS[operation][0]}', '结果名称'),
                'digest': digest,
                'parent_id': parent,
                'operation': operation,
                'parameters': parameters,
                'report': report,
                'summary': output.summary(),
                'created_at': c.timestamp(),
            }
            prepared.append(row)
            current = output
            parent = row['id']
        with self.store.transaction(project_id, expected_revision, '应用原子处理配方') as current_project:
            current_project['assets'].extend(prepared)
        return {'project': self.project_summary(project_id), 'assets': prepared, 'asset': prepared[-1]}

    def project_health(self, project_id):
        project = self.store.get(project_id)
        issues = []
        total_bytes = 0
        checked = set()
        for asset in project['assets']:
            if asset['digest'] in checked:
                continue
            checked.add(asset['digest'])
            try:
                cube = self.store.load_cube(asset['digest'])
                total_bytes += cube.values.nbytes + cube.valid.nbytes
                actual = cube.summary()
                for key in ('width', 'height', 'band_count', 'domain', 'unit'):
                    if asset['summary'].get(key) != actual[key]:
                        issues.append({'asset_id': asset['id'], 'issue': 'summary_mismatch', 'field': key})
            except c.Invalid as error:
                issues.append({'asset_id': asset['id'], 'issue': str(error)})
        for asset in project['assets']:
            try:
                self.lineage(project_id, asset['id'])
            except c.Invalid as error:
                issues.append({'asset_id': asset['id'], 'issue': str(error)})
        return {'project_id': project_id, 'revision': project['revision'], 'unique_assets_checked': len(checked), 'uncompressed_bytes': total_bytes, 'issues': issues, 'healthy': not issues}
