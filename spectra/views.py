import copy
import json
import os
import threading
from pathlib import Path

import numpy as np

from . import contracts as c
from .rendering import validate_view, default_view


class ViewPresets:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def _path(self, project_id):
        return self.root / (c.identifier(project_id) + '.json')

    def _read(self, project_id):
        path = self._path(project_id)
        if not path.exists():
            return {'revision': 0, 'presets': []}
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            raise c.Invalid('显示预设文件不可读取') from None
        c.revision(record.get('revision'))
        c.array_value(record.get('presets'), '显示预设', 0, 30)
        return record

    def _write(self, project_id, record):
        path = self._path(project_id)
        temporary = path.with_suffix('.tmp')
        with temporary.open('w') as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def list(self, project_id):
        with self.lock:
            return copy.deepcopy(self._read(project_id))

    def save(self, project_id, expected_revision, cube, name, options):
        view = validate_view(cube, options)
        bands = view['rgb'] if view['mode'] == 'rgb' else [view['band']]
        selectors = []
        for index in bands:
            band = cube.bands[index]
            selectors.append({
                'index': index,
                'name': band['name'],
                'wavelength_nm': band['wavelength_nm'],
            })
        preset = {
            'id': c.generated_id('view'),
            'name': c.text(name, '显示预设名称'),
            'created_at': c.timestamp(),
            'view': view,
            'selectors': selectors,
            'source_band_count': cube.count,
            'source_domain': cube.domain,
            'source_unit': cube.unit,
            'source_fingerprint': cube.fingerprint(),
        }
        with self.lock:
            record = self._read(project_id)
            if record['revision'] != c.revision(expected_revision):
                raise c.Conflict('显示预设已变化，请刷新')
            if len(record['presets']) >= 30:
                raise c.Invalid('工程最多30个显示预设')
            record['presets'].append(preset)
            record['revision'] += 1
            self._write(project_id, record)
            return copy.deepcopy(record)

    def remove(self, project_id, expected_revision, preset_id):
        with self.lock:
            record = self._read(project_id)
            if record['revision'] != c.revision(expected_revision):
                raise c.Conflict('显示预设已变化，请刷新')
            if not any(item['id'] == preset_id for item in record['presets']):
                raise c.Invalid('显示预设不存在')
            record['presets'] = [item for item in record['presets'] if item['id'] != preset_id]
            record['revision'] += 1
            self._write(project_id, record)
            return copy.deepcopy(record)

    def resolve(self, project_id, preset_id, cube):
        with self.lock:
            record = self._read(project_id)
            preset = next((item for item in record['presets'] if item['id'] == preset_id), None)
            if preset is None:
                raise c.Invalid('显示预设不存在')
            preset = copy.deepcopy(preset)
        warnings = []
        selected = []
        for selector in preset['selectors']:
            wavelength = selector.get('wavelength_nm')
            if wavelength is not None:
                available = [(index, abs(band['wavelength_nm'] - wavelength)) for index, band in enumerate(cube.bands) if band['wavelength_nm'] is not None]
                if not available:
                    raise c.Invalid('预设使用物理波长，但当前影像没有波长元数据')
                index, error = min(available, key=lambda item: item[1])
                if error > 1e-6:
                    raise c.Invalid('预设所需物理波长不在当前影像中；不会自动换成邻近波段')
            else:
                candidates = [index for index, band in enumerate(cube.bands) if band['name'] == selector['name']]
                if len(candidates) != 1:
                    raise c.Invalid('预设波段名称不能唯一映射至当前影像')
                index = candidates[0]
            selected.append(index)
        if cube.domain != preset['source_domain'] or cube.unit != preset['source_unit']:
            warnings.append('数值域或单位不同；只复用显示设置，不改变数值，也不保证视觉可直接比较')
        view = copy.deepcopy(preset['view'])
        if view['mode'] == 'rgb':
            view['rgb'] = selected
        else:
            view['band'] = selected[0]
        view = validate_view(cube, view)
        return {'preset_id': preset_id, 'name': preset['name'], 'view': view, 'warnings': warnings, 'mapping': selected}

    def export(self, project_id):
        record = self.list(project_id)
        return {'schema': 'materialscope_views_v1', 'presets': record['presets']}

    def import_presets(self, project_id, expected_revision, data):
        data = c.object_value(data, '显示预设包')
        if data.get('schema') != 'materialscope_views_v1':
            raise c.Invalid('显示预设包版本不支持')
        candidates = c.array_value(data.get('presets'), '显示预设', 1, 30)
        imported = []
        for candidate in candidates:
            candidate = c.object_value(candidate, '显示预设条目')
            c.text(candidate.get('name'), '显示预设名称')
            selectors = c.array_value(candidate.get('selectors'), '波段选择', 1, 3)
            for selector in selectors:
                c.text(selector.get('name'), '波段名称', 50)
                c.integer(selector.get('index'), '原波段索引', 0, 63)
                if selector.get('wavelength_nm') is not None:
                    c.number(selector['wavelength_nm'], '波长', 100, 30000)
            view = c.object_value(candidate.get('view'), '显示参数')
            mode = c.choice(view.get('mode'), '显示方式', ['rgb', 'band', 'validity', 'labels'])
            if len(selectors) != (3 if mode == 'rgb' else 1):
                raise c.Invalid('预设波段选择数量不匹配')
            c.percent_range(view.get('low'), view.get('high'))
            c.number(view.get('gamma'), '伽马', 0.1, 5)
            c.integer(view.get('maximum'), '最大预览尺寸', 64, 1600)
            c.choice(view.get('palette'), '色带', ['ocean', 'gray', 'heat', 'diverging'])
            c.choice(view.get('background'), '背景', ['checker', 'black', 'transparent'])
            row = copy.deepcopy(candidate)
            row['id'] = c.generated_id('view')
            row['created_at'] = c.timestamp()
            imported.append(row)
        with self.lock:
            record = self._read(project_id)
            if record['revision'] != c.revision(expected_revision):
                raise c.Conflict('显示预设已变化，请刷新')
            if len(record['presets']) + len(imported) > 30:
                raise c.Invalid('导入后显示预设超过30个')
            record['presets'].extend(imported)
            record['revision'] += 1
            self._write(project_id, record)
            return copy.deepcopy(record)
