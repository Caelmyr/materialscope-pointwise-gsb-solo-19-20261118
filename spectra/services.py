import copy
import hashlib
import json
import time

import numpy as np

from . import contracts as c
from . import diagnostics, library, regions, localwarp
from .demo import create_demo
from .engine import run_operation, OPERATIONS
from .io import import_cube, read_signature_csv
from .model import metadata_replacement, source_preserving
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

    # ------------------------------------------------------------------
    # 可审阅局部波段配准
    # ------------------------------------------------------------------
    def _sessions(self, project):
        return project.setdefault('registration_sessions', [])

    def _session(self, project, session_id):
        session = next((row for row in self._sessions(project) if row['id'] == session_id), None)
        if session is None:
            raise c.Invalid('局部配准审阅不存在')
        return session

    @staticmethod
    def _require_open(session):
        if session.get('status') != 'open':
            raise c.Invalid('审阅已冻结，保存的判断和结果不可修改；请复制为新审阅后再调整')

    def create_registration_session(self, project_id, asset_id, expected_revision, options):
        project, asset, cube = self.store.asset(project_id, asset_id)
        if project['revision'] != c.revision(expected_revision):
            raise c.Conflict('工程已变化，请刷新后重试')
        if len(self._sessions(project)) >= c.LIMITS['registration_sessions']:
            raise c.Invalid('局部配准审阅数量已达上限')
        reference_band = c.integer(options.get('reference_band', cube.count // 2), '参考波段', 0, cube.count - 1)
        max_displacement = c.number(options.get('max_displacement', 20), '最大局部位移像素', 0.5, min(cube.width, cube.height) / 2)
        policy = c.choice(options.get('unconstrained_policy', 'mask'), '无约束位置处理', ['mask', 'passthrough'])
        now = c.timestamp()
        session = {
            'id': c.generated_id('regsession'),
            'name': c.text(options.get('name', asset['name'] + ' · 局部配准审阅'), '审阅名称'),
            'asset_id': asset_id,
            'status': 'open',
            'reference_band': reference_band,
            'max_displacement': max_displacement,
            'unconstrained_policy': policy,
            'points': [],
            'regions': [],
            'result_asset_id': None,
            'warp_report': {},
            'created_at': now,
            'updated_at': now,
        }
        with self.store.transaction(project_id, expected_revision, '新建局部配准审阅') as current:
            self._sessions(current).append(session)
        return self.registration_session_detail(project_id, session['id'])

    def registration_session_detail(self, project_id, session_id):
        project = self.store.get(project_id)
        session = self._session(project, session_id)
        _, _, cube = self.store.asset(project_id, session['asset_id'])
        detail = copy.deepcopy(session)
        detail['revision'] = project['revision']
        detail['bands'] = list(cube.bands)
        detail['shape'] = list(cube.shape)
        detail['asset_name'] = next(row['name'] for row in project['assets'] if row['id'] == session['asset_id'])
        if session['status'] == 'open':
            analysis = localwarp.detect_conflicts(session, cube)
            detail['conflict_summary'] = {
                'blocking': analysis['blocking'],
                'conflicts': analysis['conflicts'],
                'per_band': analysis['per_band'],
            }
        return detail

    def list_registration_sessions(self, project_id):
        project = self.store.get(project_id)
        assets = {row['id']: row['name'] for row in project['assets']}
        rows = []
        for session in self._sessions(project):
            rows.append({
                'id': session['id'],
                'name': session['name'],
                'asset_id': session['asset_id'],
                'asset_name': assets.get(session['asset_id']),
                'status': session['status'],
                'reference_band': session['reference_band'],
                'point_count': len(session['points']),
                'region_count': len(session['regions']),
                'result_asset_id': session.get('result_asset_id'),
                'updated_at': session['updated_at'],
            })
        return {'revision': project['revision'], 'sessions': rows}

    def update_registration_session(self, project_id, session_id, expected_revision, options):
        project = self.store.get(project_id)
        session = self._session(project, session_id)
        self._require_open(session)
        _, _, cube = self.store.asset(project_id, session['asset_id'])
        options = c.object_value(options, '审阅修改')
        if 'reference_band' in options:
            session['reference_band'] = c.integer(options['reference_band'], '参考波段', 0, cube.count - 1)
        if 'max_displacement' in options:
            session['max_displacement'] = c.number(options['max_displacement'], '最大局部位移像素', 0.5, min(cube.width, cube.height) / 2)
        if 'unconstrained_policy' in options:
            session['unconstrained_policy'] = c.choice(options['unconstrained_policy'], '无约束位置处理', ['mask', 'passthrough'])
        if 'name' in options:
            session['name'] = c.text(options['name'], '审阅名称')
        session['updated_at'] = c.timestamp()
        with self.store.transaction(project_id, expected_revision, '更新局部配准审阅设置') as current:
            target = self._session(current, session_id)
            target.update({key: session[key] for key in ('reference_band', 'max_displacement', 'unconstrained_policy', 'name', 'updated_at')})
        return self.registration_session_detail(project_id, session_id)

    def add_registration_point(self, project_id, session_id, expected_revision, options):
        project = self.store.get(project_id)
        session = self._session(project, session_id)
        self._require_open(session)
        _, _, cube = self.store.asset(project_id, session['asset_id'])
        if len(session['points']) >= localwarp.SESSION_LIMIT_POINTS:
            raise c.Invalid('对应点数量已达上限')
        point = localwarp.validate_point(options, cube.width, cube.height)
        point['id'] = c.generated_id('regpoint')
        with self.store.transaction(project_id, expected_revision, '新增配准对应点') as current:
            target = self._session(current, session_id)
            target['points'].append(point)
            target['updated_at'] = c.timestamp()
        return self.registration_session_detail(project_id, session_id)

    def update_registration_point(self, project_id, session_id, point_id, expected_revision, options):
        project = self.store.get(project_id)
        session = self._session(project, session_id)
        self._require_open(session)
        _, _, cube = self.store.asset(project_id, session['asset_id'])
        point = next((row for row in session['points'] if row['id'] == point_id), None)
        if point is None:
            raise c.Invalid('对应点不存在')
        merged = {**point, **c.object_value(options, '对应点修改')}
        merged['id'] = point_id
        clean = localwarp.validate_point(merged, cube.width, cube.height)
        clean['id'] = point_id
        with self.store.transaction(project_id, expected_revision, '修订配准对应点') as current:
            target = self._session(current, session_id)
            index = next(i for i, row in enumerate(target['points']) if row['id'] == point_id)
            target['points'][index] = clean
            target['updated_at'] = c.timestamp()
        return self.registration_session_detail(project_id, session_id)

    def remove_registration_point(self, project_id, session_id, point_id, expected_revision):
        with self.store.transaction(project_id, expected_revision, '删除配准对应点') as current:
            session = self._session(current, session_id)
            self._require_open(session)
            before = len(session['points'])
            session['points'] = [row for row in session['points'] if row['id'] != point_id]
            if len(session['points']) == before:
                raise c.Invalid('对应点不存在')
            session['updated_at'] = c.timestamp()
        return self.registration_session_detail(project_id, session_id)

    def add_registration_region(self, project_id, session_id, expected_revision, options):
        project = self.store.get(project_id)
        session = self._session(project, session_id)
        self._require_open(session)
        _, _, cube = self.store.asset(project_id, session['asset_id'])
        if len(session['regions']) >= localwarp.SESSION_LIMIT_REGIONS:
            raise c.Invalid('材料区域数量已达上限')
        region = localwarp.validate_region(options, cube.width, cube.height)
        region['id'] = c.generated_id('regregion')
        with self.store.transaction(project_id, expected_revision, '圈定独立材料区域') as current:
            target = self._session(current, session_id)
            target['regions'].append(region)
            target['updated_at'] = c.timestamp()
        return self.registration_session_detail(project_id, session_id)

    def remove_registration_region(self, project_id, session_id, region_id, expected_revision):
        with self.store.transaction(project_id, expected_revision, '删除配准材料区域') as current:
            session = self._session(current, session_id)
            self._require_open(session)
            before = len(session['regions'])
            session['regions'] = [row for row in session['regions'] if row['id'] != region_id]
            if len(session['regions']) == before:
                raise c.Invalid('材料区域不存在')
            session['updated_at'] = c.timestamp()
        return self.registration_session_detail(project_id, session_id)

    def registration_candidates(self, project_id, session_id, options):
        project = self.store.get(project_id)
        session = self._session(project, session_id)
        self._require_open(session)
        _, _, cube = self.store.asset(project_id, session['asset_id'])
        band = c.integer(options.get('band'), '待配波段', 0, cube.count - 1)
        reference_xy = localwarp.validate_xy(options.get('reference'), cube.width, cube.height, '参考点')
        candidates = localwarp.suggest_candidates(
            cube, session['reference_band'], band, reference_xy,
            options.get('template_radius', 7), options.get('search_radius', 15),
            options.get('ambiguity_margin', 0.08),
        )
        hint = localwarp.translation_hint(cube, session['reference_band'], band, min(24, min(cube.width, cube.height) // 4))
        return {'band': band, 'reference': reference_xy, 'candidates': candidates, 'translation_hint': hint,
                'note': '候选仅辅助审阅；只有显式确认的对应点才约束配准结果'}

    def registration_conflicts(self, project_id, session_id):
        project = self.store.get(project_id)
        session = self._session(project, session_id)
        _, _, cube = self.store.asset(project_id, session['asset_id'])
        if session['status'] != 'open':
            return {'blocking': False, 'conflicts': [], 'frozen': True}
        analysis = localwarp.detect_conflicts(session, cube)
        return {'blocking': analysis['blocking'], 'conflicts': analysis['conflicts'], 'per_band': analysis['per_band']}

    def freeze_registration_session(self, project_id, session_id, expected_revision, options):
        project = self.store.get(project_id)
        session = self._session(project, session_id)
        self._require_open(session)
        _, asset, cube = self.store.asset(project_id, session['asset_id'])
        if project['revision'] != c.revision(expected_revision):
            raise c.Conflict('工程已变化，请刷新后重试')
        analysis = localwarp.detect_conflicts(session, cube)
        if analysis['blocking']:
            raise c.Invalid('存在未处理的判断冲突，已停止；请先处理冲突列表中的阻断项')
        try:
            fields = localwarp.build_warp_fields(cube, analysis)
        except c.Invalid as error:
            raise c.Invalid('局部变形存在折叠或翻转，已停止：' + str(error)) from None
        result, output_valid, band_reports = localwarp.apply_local_warp(cube, analysis, fields)
        pending_bands = [row['band'] for row in band_reports if row['status'].startswith('pending')]
        warp_summary = {
            'model': 'piecewise_affine_delaunay',
            'reference_band': session['reference_band'],
            'unconstrained_policy': session['unconstrained_policy'],
            'max_displacement_setting': session['max_displacement'],
            'point_count': len(session['points']),
            'confirmed_point_count': len([p for p in session['points'] if p['status'] == 'confirmed']),
            'pending_point_count': len([p for p in session['points'] if p['status'] == 'pending']),
            'region_count': len(session['regions']),
            'bands': band_reports,
            'pending_bands': pending_bands,
            'session_id': session_id,
            'judgment_basis': '仅由人工确认对应点与独立材料区域驱动；候选与平移提示不参与求解',
            'folding_policy': '任何三角形折叠或翻转都会阻断重算，不做折叠/翻转配准',
            'validity_policy': '严格双线性支持；缺测、越界与无约束位置不生成有效测量',
        }
        result_cube = source_preserving(cube, result, output_valid, 'local_warp_registration', warp_summary)
        digest = self.store.save_cube(result_cube)
        name = c.text(options.get('name', asset['name'] + ' · 局部配准结果'), '结果名称')
        row = {
            'id': c.generated_id('cube'),
            'name': name,
            'digest': digest,
            'parent_id': asset['id'],
            'operation': 'local_warp_registration',
            'parameters': {'session_id': session_id, 'reference_band': session['reference_band'],
                           'unconstrained_policy': session['unconstrained_policy']},
            'report': warp_summary,
            'summary': result_cube.summary(),
            'created_at': c.timestamp(),
        }
        with self.store.transaction(project_id, expected_revision, '冻结局部配准审阅并保存结果') as current:
            target = self._session(current, session_id)
            self._require_open(target)
            current['assets'].append(row)
            target['status'] = 'frozen'
            target['result_asset_id'] = row['id']
            target['warp_report'] = c.json_safe(warp_summary)
            target['updated_at'] = c.timestamp()
        return {'project': self.project_summary(project_id), 'session': self.registration_session_detail(project_id, session_id),
                'asset': row, 'report': warp_summary}

    def clone_registration_session(self, project_id, session_id, expected_revision):
        project = self.store.get(project_id)
        source = self._session(project, session_id)
        if len(self._sessions(project)) >= c.LIMITS['registration_sessions']:
            raise c.Invalid('局部配准审阅数量已达上限')
        now = c.timestamp()
        clone = copy.deepcopy(source)
        clone['id'] = c.generated_id('regsession')
        clone['name'] = c.text(source['name'] + ' · 调整副本', '审阅名称')
        clone['status'] = 'open'
        clone['result_asset_id'] = None
        clone['warp_report'] = {}
        clone['created_at'] = now
        clone['updated_at'] = now
        for point in clone['points']:
            point['id'] = c.generated_id('regpoint')
        for region in clone['regions']:
            region['id'] = c.generated_id('regregion')
        with self.store.transaction(project_id, expected_revision, '复制局部配准审阅') as current:
            self._sessions(current).append(clone)
        return self.registration_session_detail(project_id, clone['id'])

    def delete_registration_session(self, project_id, session_id, expected_revision):
        with self.store.transaction(project_id, expected_revision, '删除局部配准审阅') as current:
            session = self._session(current, session_id)
            if session.get('status') == 'frozen':
                raise c.Invalid('冻结审阅是已交付结果的判断依据，不能删除；如需调整请复制为新审阅')
            current['registration_sessions'] = [row for row in self._sessions(current) if row['id'] != session_id]
        return self.project_summary(project_id)
