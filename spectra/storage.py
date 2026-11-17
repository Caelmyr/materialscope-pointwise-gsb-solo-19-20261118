import copy
import hashlib
import json
import os
import threading
import zipfile
import io
from contextlib import contextmanager
from pathlib import Path

from . import contracts as c
from .consistency import validate_project_structure
from .io import cube_bytes, read_npz, safe_archive_entries


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.asset_root = self.root / 'assets'
        self.project_root = self.root / 'projects'
        self.asset_root.mkdir(exist_ok=True)
        self.project_root.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.cache = {}
        self.cache_order = []
        self.cache_bytes = 0
        self.cache_limit = 160000000

    def _atomic(self, path, data):
        temp = path.with_name(path.name + '.' + c.generated_id('tmp'))
        try:
            with temp.open('wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    def _project_path(self, project_id):
        return self.project_root / (c.identifier(project_id, '工程标识') + '.json')

    def _asset_path(self, digest):
        if len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
            raise c.Invalid('素材摘要无效')
        return self.asset_root / (digest + '.npz')

    def _read(self, project_id):
        path = self._project_path(project_id)
        if not path.exists():
            raise c.Invalid('工程不存在')
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            raise c.Invalid('工程数据不可读取') from None

    def _write(self, project):
        data = json.dumps(c.json_safe(project), ensure_ascii=False, indent=2, allow_nan=False).encode()
        self._atomic(self._project_path(project['id']), data)

    def list_projects(self):
        with self.lock:
            rows = []
            for path in self.project_root.glob('*.json'):
                try:
                    item = json.loads(path.read_text())
                    rows.append({
                        'id': item['id'],
                        'name': item['name'],
                        'revision': item['revision'],
                        'updated_at': item['updated_at'],
                        'asset_count': len(item['assets']),
                    })
                except (OSError, KeyError, json.JSONDecodeError):
                    continue
            return sorted(rows, key=lambda row: row['updated_at'], reverse=True)

    def create(self, name):
        with self.lock:
            project = {
                'schema': 'materialscope_project_v1',
                'id': c.generated_id('project'),
                'name': c.text(name, '工程名称'),
                'revision': 0,
                'created_at': c.timestamp(),
                'updated_at': c.timestamp(),
                'assets': [],
                'regions': [],
                'signatures': [],
                'notes': '',
                'snapshots': [],
                'history': [],
                'redo': [],
            }
            self._write(project)
            return copy.deepcopy(project)

    def get(self, project_id):
        with self.lock:
            return copy.deepcopy(self._read(project_id))

    def _state(self, project):
        return {key: copy.deepcopy(project[key]) for key in ('name', 'assets', 'regions', 'signatures', 'notes')}

    @contextmanager
    def transaction(self, project_id, expected_revision, action):
        with self.lock:
            project = self._read(project_id)
            if project['revision'] != c.revision(expected_revision):
                raise c.Conflict('工程已变化，请刷新后重试')
            before = self._state(project)
            yield project
            self.validate_project(project)
            project['history'].append({'action': action, 'time': c.timestamp(), 'state': before})
            project['history'] = project['history'][-c.LIMITS['history']:]
            project['redo'] = []
            project['revision'] += 1
            project['updated_at'] = c.timestamp()
            self._write(project)

    def validate_project(self, project):
        validate_project_structure(project)
        for key, maximum in [('assets', c.LIMITS['assets']), ('regions', c.LIMITS['regions']), ('signatures', c.LIMITS['signatures'])]:
            rows = c.array_value(project.get(key), key, 0, maximum)
            ids = [c.identifier(row.get('id'), key + '标识') for row in rows]
            if len(set(ids)) != len(ids):
                raise c.Invalid(key + '标识重复')
        assets = {row['id']: row for row in project['assets']}
        for row in project['assets']:
            if not self._asset_path(row['digest']).exists():
                raise c.Invalid('工程引用缺失的影像素材')
            if row.get('parent_id') and row['parent_id'] not in assets:
                raise c.Invalid('派生影像缺失父来源')
        for region in project['regions']:
            if region['asset_id'] not in assets:
                raise c.Invalid('分区引用的影像不存在')
        c.text(project['name'], '工程名称')
        c.text(project['notes'], '工程说明', 10000, True)

    def save_cube(self, cube):
        data = cube_bytes(cube)
        digest = hashlib.sha256(data).hexdigest()
        with self.lock:
            path = self._asset_path(digest)
            if not path.exists():
                self._atomic(path, data)
            self._cache_cube(digest, cube)
        return digest

    def _cache_cube(self, digest, cube):
        if digest in self.cache:
            self.cache_order.remove(digest)
            self.cache_order.append(digest)
            return
        size = cube.values.nbytes + cube.valid.nbytes
        if size > self.cache_limit:
            return
        while self.cache_order and self.cache_bytes + size > self.cache_limit:
            first = self.cache_order.pop(0)
            removed = self.cache.pop(first)
            self.cache_bytes -= removed.values.nbytes + removed.valid.nbytes
        self.cache[digest] = cube
        self.cache_order.append(digest)
        self.cache_bytes += size

    def load_cube(self, digest):
        with self.lock:
            if digest in self.cache:
                self.cache_order.remove(digest)
                self.cache_order.append(digest)
                return self.cache[digest]
            path = self._asset_path(digest)
            if not path.exists():
                raise c.Invalid('影像素材不存在')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise c.Invalid('影像素材校验失败')
            cube = read_npz(data)
            self._cache_cube(digest, cube)
            return cube

    def asset(self, project_id, asset_id):
        project = self.get(project_id)
        asset_id = c.identifier(asset_id, '影像标识')
        row = next((row for row in project['assets'] if row['id'] == asset_id), None)
        if row is None:
            raise c.Invalid('影像不存在')
        return project, row, self.load_cube(row['digest'])

    def append_cube(self, project_id, expected_revision, cube, name, parent_id=None, operation='import', parameters=None, report=None):
        digest = self.save_cube(cube)
        row = {
            'id': c.generated_id('cube'),
            'name': c.text(name, '影像名称'),
            'digest': digest,
            'parent_id': parent_id,
            'operation': operation,
            'parameters': c.options_copy(parameters or {}),
            'report': c.json_safe(report or {}),
            'summary': cube.summary(),
            'created_at': c.timestamp(),
        }
        with self.transaction(project_id, expected_revision, '新增影像：' + row['name']) as project:
            project['assets'].append(row)
        return self.get(project_id), row

    def update_details(self, project_id, expected_revision, options):
        with self.transaction(project_id, expected_revision, '更新工程说明') as project:
            if 'name' in options:
                project['name'] = c.text(options['name'], '工程名称')
            if 'notes' in options:
                project['notes'] = c.text(options['notes'], '说明', 10000, True)
        return self.get(project_id)

    def rename_asset(self, project_id, expected_revision, asset_id, name):
        with self.transaction(project_id, expected_revision, '重命名影像') as project:
            asset = next((row for row in project['assets'] if row['id'] == asset_id), None)
            if asset is None:
                raise c.Invalid('影像不存在')
            asset['name'] = c.text(name, '影像名称')
        return self.get(project_id)

    def remove_asset(self, project_id, expected_revision, asset_id):
        with self.transaction(project_id, expected_revision, '移除叶影像') as project:
            if any(row.get('parent_id') == asset_id for row in project['assets']):
                raise c.Invalid('该影像仍有派生结果，请先移除派生结果')
            if not any(row['id'] == asset_id for row in project['assets']):
                raise c.Invalid('影像不存在')
            project['assets'] = [row for row in project['assets'] if row['id'] != asset_id]
            project['regions'] = [row for row in project['regions'] if row['asset_id'] != asset_id]
        return self.get(project_id)

    def undo(self, project_id, expected_revision, direction='undo'):
        with self.lock:
            project = self._read(project_id)
            if project['revision'] != c.revision(expected_revision):
                raise c.Conflict('工程已变化，请刷新')
            source_key = 'history' if direction == 'undo' else 'redo'
            target_key = 'redo' if direction == 'undo' else 'history'
            if not project[source_key]:
                raise c.Invalid('没有可' + ('撤销' if direction == 'undo' else '重做') + '的修改')
            restored = project[source_key].pop()
            project[target_key].append({'action': restored['action'], 'time': c.timestamp(), 'state': self._state(project)})
            project.update(restored['state'])
            self.validate_project(project)
            project['revision'] += 1
            project['updated_at'] = c.timestamp()
            self._write(project)
            return copy.deepcopy(project)

    def snapshot(self, project_id, expected_revision, name):
        with self.transaction(project_id, expected_revision, '保存命名版本') as project:
            if len(project['snapshots']) >= 20:
                raise c.Invalid('最多20个命名版本')
            project['snapshots'].append({
                'id': c.generated_id('snapshot'),
                'name': c.text(name, '版本名称'),
                'created_at': c.timestamp(),
                'state': self._state(project),
            })
        return self.get(project_id)

    def restore_snapshot(self, project_id, expected_revision, snapshot_id):
        with self.transaction(project_id, expected_revision, '恢复命名版本') as project:
            snapshot = next((row for row in project['snapshots'] if row['id'] == snapshot_id), None)
            if snapshot is None:
                raise c.Invalid('命名版本不存在')
            project.update(copy.deepcopy(snapshot['state']))
        return self.get(project_id)

    def archive(self, project_id):
        project = self.get(project_id)
        digests = set()
        for state in [project] + [item['state'] for item in project['snapshots']]:
            digests.update(row['digest'] for row in state['assets'])
        project['history'] = []
        project['redo'] = []
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('project.json', json.dumps(project, ensure_ascii=False))
            for digest in sorted(digests):
                data = self._asset_path(digest).read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise c.Invalid('素材校验失败，不能导出工程')
                archive.writestr('assets/' + digest + '.npz', data)
        if output.tell() > c.LIMITS['archive_bytes']:
            raise c.Invalid('工程导出超过120MB')
        return output.getvalue()

    def import_archive(self, data):
        with safe_archive_entries(data) as archive:
            try:
                project = json.loads(archive.read('project.json'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise c.Invalid('工程描述无效') from None
            if project.get('schema') != 'materialscope_project_v1':
                raise c.Invalid('不支持的工程版本')
            c.object_value(project, '工程')
            states = [project] + [item['state'] for item in project.get('snapshots', [])]
            digests = {row['digest'] for state in states for row in state.get('assets', [])}
            prepared = {}
            for digest in digests:
                self._asset_path(digest)
                member = 'assets/' + digest + '.npz'
                if member not in archive.namelist():
                    raise c.Invalid('工程缺少素材：' + digest[:12])
                asset_data = archive.read(member)
                if hashlib.sha256(asset_data).hexdigest() != digest:
                    raise c.Invalid('工程素材摘要不一致')
                prepared[digest] = (asset_data, read_npz(asset_data))
            with self.lock:
                for digest, (asset_data, cube) in prepared.items():
                    if not self._asset_path(digest).exists():
                        self._atomic(self._asset_path(digest), asset_data)
                    self._cache_cube(digest, cube)
                self.validate_project(project)
                for snapshot in project.get('snapshots', []):
                    snapshot_state = {**project, **snapshot['state']}
                    self.validate_project(snapshot_state)
                project['id'] = c.generated_id('project')
                project['name'] = c.text(project['name'] + ' · 导入', '工程名称')
                project['revision'] = 0
                project['created_at'] = c.timestamp()
                project['updated_at'] = c.timestamp()
                project['history'] = []
                project['redo'] = []
                self._write(project)
        return copy.deepcopy(project)

    def cache_status(self):
        with self.lock:
            return {'entries': len(self.cache), 'bytes': self.cache_bytes, 'limit': self.cache_limit}
