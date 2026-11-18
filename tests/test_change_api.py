"""变化分析端到端 API 流程测试（Flask 测试客户端，临时数据目录）。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra import create_app


def main():
    tmp = tempfile.mkdtemp()
    app = create_app(tmp)
    client = app.test_client()

    # 装载变化示范
    r = client.post('/api/change-demo', json={})
    assert r.status_code == 200, r.data
    demo = r.get_json()
    pid = demo['project']['id']
    before_ids = demo['before_asset_ids']
    after_ids = demo['after_asset_ids']
    print('demo project', pid, 'before', len(before_ids), 'after', len(after_ids))

    # 两取向预览（同一份证据）
    r = client.post(f'/api/projects/{pid}/change/preview', json={
        'before_asset_ids': before_ids, 'after_asset_ids': after_ids, 'options': {}})
    assert r.status_code == 200, r.data
    preview = r.get_json()
    assert set(preview['policies']) == {'formal', 'screening'}
    for pol in ('formal', 'screening'):
        pa = preview['policies'][pol]['pixel_accounting']
        print('preview', pol, 'confirmed', pa['confirmed_change_pixels'],
              'undecided', pa['undecided_pixels'], 'tested', pa['tested_pixels'])
    formal_pa = preview['policies']['formal']['pixel_accounting']
    screen_pa = preview['policies']['screening']['pixel_accounting']
    # 筛查取向候选不少于正式取向
    assert screen_pa['confirmed_change_pixels'] >= formal_pa['confirmed_change_pixels']
    assert len(preview['policies']['formal']['label_preview']) > 0

    # 无效取向应被拒绝
    r = client.post(f'/api/projects/{pid}/change/confirm', json={
        'before_asset_ids': before_ids, 'after_asset_ids': after_ids,
        'policy': 'bogus', 'revision': demo['project']['revision']})
    assert r.status_code == 400
    print('invalid policy rejected:', r.get_json()['error'][:30])

    # 网格不一致应被拒绝（用单个影像凑数组 -> 同网格其实通过；改用错误长度）
    r = client.post(f'/api/projects/{pid}/change/preview', json={
        'before_asset_ids': before_ids[:1], 'after_asset_ids': after_ids, 'options': {}})
    assert r.status_code == 400
    print('too few repeats rejected')

    # 确认正式取向（先取最新修订号，装载示范内部经过多次事务）
    revision = client.get(f'/api/projects/{pid}').get_json()['project']['revision']
    r = client.post(f'/api/projects/{pid}/change/confirm', json={
        'before_asset_ids': before_ids, 'after_asset_ids': after_ids,
        'policy': 'formal', 'revision': revision, 'name': '正式变化分析'})
    assert r.status_code == 200, r.data
    confirmed = r.get_json()
    analysis = confirmed['change_analysis']
    aid = analysis['id']
    print('confirmed analysis', aid, 'assets', [a['operation'] for a in confirmed['assets']])
    assert analysis['policy'] == 'formal'
    assert analysis['protocol']['before']['repeat_count'] == 6
    assert len(analysis['protocol']['before']['asset_fingerprints']) == 6

    # 列表
    r = client.get(f'/api/projects/{pid}/change-analyses')
    assert r.status_code == 200
    assert len(r.get_json()['change_analyses']) == 1

    # 点选证据
    r = client.get(f'/api/projects/{pid}/change-analyses/{aid}/evidence?x=40&y=80')
    assert r.status_code == 200, r.data
    ev = r.get_json()
    assert len(ev['before_repeats']) == 6 and len(ev['after_repeats']) == 6
    assert ev['common_comparable_band_count'] >= 1
    assert ev['brightness_alignment'].startswith('none')
    print('evidence at (40,80): comparable bands', ev['common_comparable_band_count'],
          'delta sample', [None if v is None else round(v, 3) for v in ev['delta_after_minus_before'][:4]])

    # 导出：幅度 NPZ、决策 CSV、两个掩模 PNG
    for kind, ctype in [('magnitude', 'octet'), ('decision', 'csv'),
                        ('confirmed', 'png'), ('undecided', 'png')]:
        r = client.get(f'/api/projects/{pid}/change-analyses/{aid}/export/{kind}')
        assert r.status_code == 200, (kind, r.data[:100])
        assert len(r.data) > 0
    print('all four exports OK')

    # 修订冲突保护
    r = client.post(f'/api/projects/{pid}/change/confirm', json={
        'before_asset_ids': before_ids, 'after_asset_ids': after_ids,
        'policy': 'screening', 'revision': revision, 'name': '过期修订'})
    assert r.status_code == 409
    print('stale revision rejected with 409')

    # 重新打开（新 app 实例读同一目录），验证固化与依赖核验
    app2 = create_app(tempfile.mkdtemp())  # 不同目录应看不到
    r = app2.test_client().get(f'/api/projects/{pid}/change-analyses')
    # 新目录无此工程 -> 404/错误
    assert r.status_code in (400, 404)
    app_reopen = create_app(tmp)
    client2 = app_reopen.test_client()
    r = client2.get(f'/api/projects/{pid}/change-analyses')
    assert r.status_code == 200
    reopened = r.get_json()['change_analyses'][0]
    assert reopened['id'] == aid
    assert reopened['protocol']['analysis_band_indices'] == analysis['protocol']['analysis_band_indices']
    # 重开后仍可回看证据（指纹可解析）
    r = client2.get(f'/api/projects/{pid}/change-analyses/{aid}/evidence?x=10&y=10')
    assert r.status_code == 200
    print('reopen: analysis fixed, groups/bands/policy/evidence intact')

    # 工程 ZIP 归档包含变化分析记录
    r = client2.get(f'/api/projects/{pid}/archive')
    assert r.status_code == 200 and len(r.data) > 1000
    import zipfile, io
    with zipfile.ZipFile(io.BytesIO(r.data)) as z:
        import json
        proj = json.loads(z.read('project.json'))
        assert len(proj['change_analyses']) == 1
        # 结果影像素材已归档
        needed = {a['digest'] for a in proj['assets'] if a['operation'].startswith('change_')}
        archived = {n.split('/')[-1].replace('.npz', '') for n in z.namelist() if n.startswith('assets/')}
        assert needed.issubset(archived)
    print('archive contains change analysis and its cubes')

    # 归档导入往返：变化分析记录随工程保留
    reimport = client2.post('/api/archive',
                            data={'file': (io.BytesIO(r.data), 'roundtrip.zip')},
                            content_type='multipart/form-data')
    assert reimport.status_code == 200
    new_pid = reimport.get_json()['project']['id']
    assert len(client2.get(f'/api/projects/{new_pid}/change-analyses').get_json()['change_analyses']) == 1
    print('archive reimport preserves change analysis')

    print('\nALL CHANGE-ANALYSIS API TESTS PASSED')


if __name__ == '__main__':
    main()
