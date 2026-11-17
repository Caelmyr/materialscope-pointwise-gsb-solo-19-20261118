import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import contracts as c


class JobManager:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='materialscope')
        self.futures = {}
        self.records = {}
        self.max_pending = 8
        self._recover()

    def _recover(self):
        for path in self.root.glob('*.json'):
            try:
                record = json.loads(path.read_text())
                if record.get('state') in ('queued', 'running'):
                    record['state'] = 'interrupted'
                    record['finished_at'] = c.timestamp()
                    record['message'] = '服务重启，原任务未自动重放；请核对工程后决定是否重新执行'
                    self._save(record)
                self.records[record['id']] = record
            except (OSError, json.JSONDecodeError, KeyError):
                continue

    def _save(self, record):
        path = self.root / (c.identifier(record['id']) + '.json')
        temporary = path.with_suffix('.tmp')
        with temporary.open('w') as handle:
            json.dump(c.json_safe(record), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def submit(self, project_id, description, function, arguments):
        with self.lock:
            active = sum(row['state'] in ('queued', 'running') for row in self.records.values())
            if active >= self.max_pending:
                raise c.Invalid('最多8个活动任务，请等待已有任务结束')
            job = {
                'id': c.generated_id('job'),
                'project_id': c.identifier(project_id),
                'description': c.text(description, '任务说明', 200),
                'state': 'queued',
                'created_at': c.timestamp(),
                'started_at': None,
                'finished_at': None,
                'elapsed_seconds': None,
                'message': '等待单工作线程',
                'result': None,
            }
            self.records[job['id']] = job
            self._save(job)
            future = self.executor.submit(self._execute, job['id'], function, arguments)
            self.futures[job['id']] = future
            return copy.deepcopy(job)

    def _execute(self, job_id, function, arguments):
        with self.lock:
            record = self.records[job_id]
            if record['state'] == 'cancelled':
                return
            record['state'] = 'running'
            record['started_at'] = c.timestamp()
            record['message'] = '计算中；提交时仍核对原工程修订号'
            self._save(record)
        started = time.monotonic()
        try:
            result = function(**arguments)
            state = 'completed'
            message = '计算与工程提交完成'
            if isinstance(result, dict):
                asset = result.get('asset')
                summary = {'asset_id': asset['id'] if asset else None}
                if result.get('assets'):
                    summary['asset_ids'] = [item['id'] for item in result['assets']]
                if result.get('project'):
                    summary['revision'] = result['project']['revision']
            else:
                summary = None
        except c.Conflict as error:
            state = 'conflict'
            message = str(error)
            summary = None
        except c.Invalid as error:
            state = 'failed'
            message = str(error)
            summary = None
        except Exception as error:
            state = 'failed'
            message = type(error).__name__ + ': ' + str(error)[:300]
            summary = None
        with self.lock:
            record = self.records[job_id]
            record['state'] = state
            record['message'] = message
            record['result'] = summary
            record['elapsed_seconds'] = time.monotonic() - started
            record['finished_at'] = c.timestamp()
            self._save(record)

    def list(self, project_id=None):
        with self.lock:
            rows = list(self.records.values())
            if project_id:
                rows = [row for row in rows if row['project_id'] == project_id]
            rows.sort(key=lambda row: row['created_at'], reverse=True)
            return copy.deepcopy(rows[:100])

    def get(self, job_id):
        with self.lock:
            if job_id not in self.records:
                raise c.Invalid('任务不存在')
            return copy.deepcopy(self.records[job_id])

    def cancel(self, job_id):
        with self.lock:
            if job_id not in self.records:
                raise c.Invalid('任务不存在')
            record = self.records[job_id]
            if record['state'] != 'queued':
                raise c.Invalid('只可取消尚未开始的任务；运行中的完整计算保留真实结果')
            future = self.futures.get(job_id)
            if future is None or not future.cancel():
                raise c.Invalid('任务刚刚开始，不能取消')
            record['state'] = 'cancelled'
            record['message'] = '用户取消尚未开始的任务'
            record['finished_at'] = c.timestamp()
            self._save(record)
            return copy.deepcopy(record)

    def purge_finished(self, project_id):
        with self.lock:
            removed = []
            for job_id, record in list(self.records.items()):
                if record['project_id'] != project_id or record['state'] in ('queued', 'running'):
                    continue
                path = self.root / (job_id + '.json')
                if path.exists():
                    path.unlink()
                removed.append(job_id)
                self.records.pop(job_id)
                self.futures.pop(job_id, None)
            return {'removed_job_ids': removed}

    def status(self):
        with self.lock:
            counts = {}
            for row in self.records.values():
                counts[row['state']] = counts.get(row['state'], 0) + 1
            return {'counts': counts, 'max_pending': self.max_pending, 'workers': 1}
