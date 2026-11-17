import io
import json
import os
import traceback

from flask import Flask, jsonify, request, send_file, render_template
from werkzeug.exceptions import HTTPException

from . import contracts as c
from . import acquisition, diagnostics, geometry, library, regions, reports, repeats, scanlines, consistency
from .demo import demo_description
from .engine import operation_catalog
from .io import cube_bytes, band_tiff, mask_png, csv_bytes, spectrum_csv, result_bundle
from .jobs import JobManager
from .preflight import preflight_operation, recipe_preflight
from .rendering import PreviewCache, render_png, default_view, overlay_regions, band_contact_sheet, compare_preview
from .services import Services
from .storage import Store
from .views import ViewPresets


def create_app(data_dir):
    app = Flask(__name__, static_folder='../static', template_folder='../templates')
    app.config['MAX_CONTENT_LENGTH'] = c.LIMITS['archive_bytes']
    app.json.ensure_ascii = False
    store = Store(data_dir)
    service = Services(store)
    cache = PreviewCache()
    views = ViewPresets(os.path.join(data_dir, "views"))
    jobs = JobManager(os.path.join(data_dir, 'jobs'))
    app.extensions['materialscope'] = {'store': store, 'service': service, 'jobs': jobs, 'cache': cache}

    def payload():
        data = request.get_json(silent=True)
        return c.object_value(data, '请求JSON')

    def response(value):
        return jsonify(c.json_safe(value))

    def binary(data, filename, mimetype):
        return send_file(io.BytesIO(data), download_name=filename, mimetype=mimetype, as_attachment=request.args.get('download', '1') == '1')

    def load_asset(project_id, asset_id):
        return store.asset(project_id, asset_id)

    def view_options(cube):
        raw = request.args.get('view')
        if raw:
            try:
                return c.object_value(json.loads(raw), '显示参数')
            except json.JSONDecodeError:
                raise c.Invalid('显示参数不是有效JSON') from None
        return default_view(cube)

    def region_option(project_id):
        region_id = request.args.get('region_id')
        if not region_id:
            return None
        region, cube = service.region(project_id, region_id)
        return region

    @app.errorhandler(c.Conflict)
    def conflict(error):
        return response({'error': str(error), 'kind': 'revision_conflict'}), 409

    @app.errorhandler(c.Invalid)
    def invalid(error):
        return response({'error': str(error), 'kind': 'invalid_input'}), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return response({'error': error.description, 'kind': 'http_error'}), error.code

    @app.errorhandler(Exception)
    def unexpected(error):
        app.logger.exception('MaterialScope request failed')
        return response({'error': '处理失败：' + type(error).__name__ + ': ' + str(error)[:200], 'kind': 'processing_error'}), 500

    @app.get('/')
    def home():
        return render_template('index.html')

    @app.get('/api/health')
    def health():
        return response({'status': 'ok', 'product': 'MaterialScope', 'schema': 1, 'limits': c.LIMITS, 'cube_cache': store.cache_status(), 'preview_cache': cache.status(), 'jobs': jobs.status()})

    @app.get('/api/catalog')
    def catalog():
        return response({'operations': operation_catalog(), 'demo': demo_description(), 'acquisition_template': acquisition.acquisition_template()})

    @app.get('/api/projects')
    def projects():
        return response({'projects': store.list_projects()})

    @app.post('/api/projects')
    def create_project():
        data = payload()
        project = store.create(data.get('name', '材料分析工程'))
        return response({'project': service.project_summary(project['id'])})

    @app.post('/api/demo')
    def load_demo():
        data = payload()
        return response(service.load_demo(data.get('name', '合成材料试片实验')))

    @app.get('/api/projects/<project_id>')
    def project(project_id):
        return response({'project': service.project_summary(project_id)})

    @app.patch('/api/projects/<project_id>')
    def update_project(project_id):
        data = payload()
        store.update_details(project_id, data.get('revision'), data)
        return response({'project': service.project_summary(project_id)})

    @app.get('/api/projects/<project_id>/health')
    def project_health(project_id):
        return response(service.project_health(project_id))

    @app.post('/api/projects/<project_id>/history/<direction>')
    def history(project_id, direction):
        c.choice(direction, '历史方向', ['undo', 'redo'])
        data = payload()
        store.undo(project_id, data.get('revision'), direction)
        return response({'project': service.project_summary(project_id)})

    @app.post('/api/projects/<project_id>/snapshots')
    def snapshot(project_id):
        data = payload()
        store.snapshot(project_id, data.get('revision'), data.get('name'))
        return response({'project': service.project_summary(project_id)})

    @app.post('/api/projects/<project_id>/snapshots/<snapshot_id>/restore')
    def restore_snapshot(project_id, snapshot_id):
        data = payload()
        store.restore_snapshot(project_id, data.get('revision'), snapshot_id)
        return response({'project': service.project_summary(project_id)})

    @app.get('/api/projects/<project_id>/archive')
    def export_project(project_id):
        return binary(store.archive(project_id), 'materialscope-project.zip', 'application/zip')

    @app.post('/api/archive')
    def import_project():
        upload = request.files.get('file')
        if upload is None:
            raise c.Invalid('请上传工程ZIP')
        project = store.import_archive(upload.read())
        return response({'project': service.project_summary(project['id'])})

    @app.get('/api/projects/<project_id>/dependencies')
    def dependencies(project_id):
        return response(consistency.project_dependencies(store.get(project_id)))

    @app.get('/api/projects/<project_id>/snapshots/<snapshot_id>/compare')
    def snapshot_compare(project_id, snapshot_id):
        return response(consistency.compare_snapshot(store.get(project_id), snapshot_id))

    @app.get('/api/projects/<project_id>/inventory.csv')
    def inventory(project_id):
        return binary(csv_bytes(reports.project_inventory(store.get(project_id))), 'project-inventory.csv', 'text/csv')

    @app.post('/api/projects/<project_id>/assets')
    def import_asset(project_id):
        upload = request.files.get('file')
        if upload is None:
            raise c.Invalid('请上传图像或NPZ')
        return response(service.import_asset(project_id, request.form.get('revision'), upload.read(), upload.filename, request.form.get('name') or None))

    @app.post('/api/projects/<project_id>/acquisition/<kind>')
    def import_acquisition(project_id, kind):
        c.choice(kind, '采集类型', ['raw', 'stack'])
        upload = request.files.get('file')
        if upload is None:
            raise c.Invalid('请上传采集文件')
        data = upload.read()
        if kind == 'raw':
            try:
                options = json.loads(request.form.get('options', '{}'))
            except json.JSONDecodeError:
                raise c.Invalid('采集布局不是有效JSON') from None
            cube, report = acquisition.decode_raw(data, options)
        else:
            cube, report = acquisition.decode_band_stack(data)
        _, row = store.append_cube(project_id, request.form.get('revision'), cube, request.form.get('name') or upload.filename, operation='acquisition_' + kind, report=report)
        return response({'project': service.project_summary(project_id), 'asset': row})

    @app.get('/api/projects/<project_id>/assets/<asset_id>')
    def asset(project_id, asset_id):
        return response(service.asset_detail(project_id, asset_id))

    @app.patch('/api/projects/<project_id>/assets/<asset_id>')
    def rename_asset(project_id, asset_id):
        data = payload()
        store.rename_asset(project_id, data.get('revision'), asset_id, data.get('name'))
        return response({'project': service.project_summary(project_id)})

    @app.delete('/api/projects/<project_id>/assets/<asset_id>')
    def delete_asset(project_id, asset_id):
        data = payload()
        store.remove_asset(project_id, data.get('revision'), asset_id)
        return response({'project': service.project_summary(project_id)})

    @app.post('/api/projects/<project_id>/assets/<asset_id>/metadata')
    def metadata(project_id, asset_id):
        data = payload()
        return response(service.modify_metadata(project_id, asset_id, data.get('revision'), data.get('options', {})))

    @app.get('/api/projects/<project_id>/assets/<asset_id>/preview.png')
    def preview(project_id, asset_id):
        _, _, cube = load_asset(project_id, asset_id)
        return send_file(io.BytesIO(render_png(cube, view_options(cube), cache)), mimetype='image/png')

    @app.get('/api/projects/<project_id>/assets/<asset_id>/overlay.png')
    def overlay(project_id, asset_id):
        project, _, cube = load_asset(project_id, asset_id)
        selected = [row for row in project['regions'] if row['asset_id'] == asset_id]
        return send_file(io.BytesIO(overlay_regions(cube, view_options(cube), selected)), mimetype='image/png')

    @app.get('/api/projects/<project_id>/assets/<asset_id>/contact-sheet.png')
    def contact_sheet(project_id, asset_id):
        _, _, cube = load_asset(project_id, asset_id)
        options = {'columns': request.args.get('columns', 4), 'tile_size': request.args.get('tile_size', 180)}
        return binary(band_contact_sheet(cube, options), 'band-contact-sheet.png', 'image/png')

    @app.get('/api/projects/<project_id>/assets/<asset_id>/pixel')
    def pixel(project_id, asset_id):
        _, _, cube = load_asset(project_id, asset_id)
        x = request.args.get('x', 0)
        y = request.args.get('y', 0)
        radius = request.args.get('radius', 0)
        if int(radius) == 0:
            result = cube.pixel(x, y)
        else:
            result = regions.neighborhood_spectrum(cube, x, y, radius)
        return response(result)

    @app.post('/api/projects/<project_id>/assets/<asset_id>/profile')
    def profile(project_id, asset_id):
        data = payload()
        _, _, cube = load_asset(project_id, asset_id)
        return response(regions.line_profile(cube, data.get('start'), data.get('end'), data.get('band', 0), data.get('samples', 200)))

    @app.get('/api/projects/<project_id>/assets/<asset_id>/diagnostics/<kind>')
    def diagnostic(project_id, asset_id, kind):
        _, _, cube = load_asset(project_id, asset_id)
        if kind == 'validity':
            result = diagnostics.validity_report(cube)
        elif kind == 'wavelengths':
            result = diagnostics.spectral_spacing_report(cube)
        elif kind == 'metadata':
            result = diagnostics.metadata_audit(cube)
        elif kind == 'capabilities':
            result = diagnostics.processing_capabilities(cube)
        elif kind == 'band':
            result = diagnostics.band_quality_report(cube, request.args.get('band', 0))
        elif kind == 'uniformity':
            result = diagnostics.reference_uniformity(cube, region_option(project_id))
        elif kind == 'correlation':
            result = regions.correlation_report(cube, region_option(project_id))
        elif kind == 'scanlines':
            result = scanlines.scanline_statistics(cube, dict(request.args))
        elif kind == 'frequency':
            result = scanlines.frequency_diagnostic(cube, dict(request.args))
        elif kind == 'pair':
            result = diagnostics.pairwise_band_difference(cube, request.args.get('first', 0), request.args.get('second', min(1, cube.count - 1)))
        else:
            raise c.Invalid('未知诊断类型')
        return response(result)

    @app.get('/api/projects/<project_id>/assets/<asset_id>/lineage')
    def lineage(project_id, asset_id):
        return response(service.lineage(project_id, asset_id))

    @app.post('/api/projects/<project_id>/assets/<asset_id>/preflight')
    def preflight(project_id, asset_id):
        data = payload()
        project, _, cube = load_asset(project_id, asset_id)
        return response(preflight_operation(cube, data.get('operation'), data.get('options', {}), project))

    @app.post('/api/projects/<project_id>/assets/<asset_id>/process')
    def process(project_id, asset_id):
        data = payload()
        arguments = {'project_id': project_id, 'asset_id': asset_id, 'expected_revision': data.get('revision'), 'operation': data.get('operation'), 'options': data.get('options', {}), 'name': data.get('name')}
        if data.get('background', False):
            job = jobs.submit(project_id, '处理影像：' + str(data.get('operation')), service.execute, arguments)
            return response({'job': job}), 202
        return response(service.execute(**arguments))

    @app.post('/api/projects/<project_id>/assets/<asset_id>/recipe')
    def recipe(project_id, asset_id):
        data = payload()
        if data.get('preflight'):
            project, _, cube = load_asset(project_id, asset_id)
            return response(recipe_preflight(cube, data.get('steps'), project))
        arguments = {'project_id': project_id, 'asset_id': asset_id, 'expected_revision': data.get('revision'), 'steps': data.get('steps'), 'name': data.get('name', '处理配方')}
        if data.get('background', False):
            return response({'job': jobs.submit(project_id, '执行处理配方', service.execute_recipe, arguments)}), 202
        return response(service.execute_recipe(**arguments))

    @app.post('/api/projects/<project_id>/assets/<asset_id>/regions')
    def create_region(project_id, asset_id):
        data = payload()
        return response(service.create_region(project_id, asset_id, data.get('revision'), data.get('region', {})))

    @app.get('/api/projects/<project_id>/regions/<region_id>')
    def region(project_id, region_id):
        return response(service.region_report(project_id, region_id))

    @app.patch('/api/projects/<project_id>/regions/<region_id>')
    def update_region(project_id, region_id):
        data = payload()
        return response(service.update_region(project_id, region_id, data.get('revision'), data.get('region', {})))

    @app.delete('/api/projects/<project_id>/regions/<region_id>')
    def remove_region(project_id, region_id):
        data = payload()
        return response({'project': service.delete_region(project_id, region_id, data.get('revision'))})

    @app.get('/api/projects/<project_id>/regions/<region_id>/quality')
    def region_quality(project_id, region_id):
        region, cube = service.region(project_id, region_id)
        return response(regions.region_quality(cube, region))

    @app.post('/api/projects/<project_id>/regions/<region_id>/signature')
    def region_signature(project_id, region_id):
        data = payload()
        return response(service.save_region_signature(project_id, region_id, data.get('revision'), data.get('name')))

    @app.post('/api/projects/<project_id>/regions/compare')
    def compare_regions(project_id):
        data = payload()
        selected = c.array_value(data.get('region_ids'), '分区', 2, 20)
        items = [service.region(project_id, item) for item in selected]
        if len({item[0]['asset_id'] for item in items}) != 1:
            raise c.Invalid('分区比较要求来自同一影像')
        return response(regions.compare_regions(items[0][1], [item[0] for item in items]))

    @app.post('/api/projects/<project_id>/signatures')
    def import_signature(project_id):
        data = payload()
        return response(service.import_signature(project_id, data.get('revision'), data.get('signature', {})))

    @app.post('/api/projects/<project_id>/signatures/csv')
    def import_signature_csv(project_id):
        upload = request.files.get('file')
        if upload is None:
            raise c.Invalid('请上传光谱CSV')
        return response(service.import_signature_csv(project_id, request.form.get('revision'), upload.read(), request.form.get('name') or upload.filename))

    @app.delete('/api/projects/<project_id>/signatures/<signature_id>')
    def delete_signature(project_id, signature_id):
        data = payload()
        return response({'project': service.delete_signature(project_id, data.get('revision'), signature_id)})

    @app.post('/api/projects/<project_id>/signatures/compare')
    def compare_signatures(project_id):
        data = payload()
        return response(service.compare_signatures(project_id, data.get('signature_ids')))

    @app.get('/api/projects/<project_id>/signatures.csv')
    def signatures_csv(project_id):
        return binary(csv_bytes(reports.signature_table(store.get(project_id)['signatures'])), 'material-library.csv', 'text/csv')

    @app.post('/api/projects/<project_id>/compare')
    def compare(project_id):
        data = payload()
        return response(service.compare_assets(project_id, data.get('first_id'), data.get('second_id'), data))

    @app.post('/api/projects/<project_id>/compare-preview')
    def comparison_preview(project_id):
        data = payload()
        _, _, first = load_asset(project_id, data.get('first_id'))
        _, _, second = load_asset(project_id, data.get('second_id'))
        return send_file(io.BytesIO(compare_preview(first, second, data.get('options', {}))), mimetype='image/png')

    @app.post('/api/projects/<project_id>/repeats')
    def repeat_analysis(project_id):
        data = payload()
        selected = c.array_value(data.get('asset_ids'), '重复测量影像', 2, 12)
        loaded = [load_asset(project_id, item) for item in selected]
        cubes = [item[2] for item in loaded]
        if data.get('diagnostic', False):
            return response(repeats.repeat_stability(cubes))
        combined, uncertainty, report = repeats.combine_repeated(cubes, data.get('options', {}))
        parent = selected[0]
        prepared = []
        for cube, suffix, operation in [(combined, '汇总', 'repeat_combine'), (uncertainty, '标准误差', 'repeat_standard_error')]:
            row = {
                'id': c.generated_id('cube'),
                'name': c.text(data.get('name', '重复测量') + ' · ' + suffix, '结果名称'),
                'digest': store.save_cube(cube),
                'parent_id': parent,
                'operation': operation,
                'parameters': {'asset_ids': selected, **data.get('options', {})},
                'report': report,
                'summary': cube.summary(),
                'created_at': c.timestamp(),
            }
            prepared.append(row)
        with store.transaction(project_id, data.get('revision'), '汇总重复测量') as project:
            project['assets'].extend(prepared)
        return response({'project': service.project_summary(project_id), 'assets': prepared, 'asset': prepared[0]})

    @app.get('/api/projects/<project_id>/assets/<asset_id>/export/<kind>')
    def export_asset(project_id, asset_id, kind):
        project, asset, cube = load_asset(project_id, asset_id)
        band = request.args.get('band', 0)
        if kind == 'bundle':
            return binary(result_bundle(cube, asset['name'], asset['report']), 'portable-result.zip', 'application/zip')
        if kind == 'pixel.csv':
            sample = cube.pixel(request.args.get('x', 0), request.args.get('y', 0))
            return binary(spectrum_csv(sample), 'pixel-spectrum.csv', 'text/csv')
        if kind == 'npz':
            return binary(cube_bytes(cube), 'cube.npz', 'application/octet-stream')
        if kind == 'tiff':
            return binary(band_tiff(cube, band), 'band.tiff', 'image/tiff')
        if kind == 'mask':
            return binary(mask_png(cube, band if 'band' in request.args else None), 'valid.png', 'image/png')
        if kind == 'preview':
            return binary(render_png(cube, view_options(cube)), 'preview.png', 'image/png')
        if kind == 'stack':
            return binary(reports.export_band_stack(cube), 'band-stack.zip', 'application/zip')
        if kind == 'raw':
            return binary(reports.export_raw(cube, {'interleave': request.args.get('interleave', 'BIL')}), 'raw-capture.zip', 'application/zip')
        if kind == 'report':
            return binary(reports.report_bundle(project, asset, cube, {'view': view_options(cube)}), 'measurement-report.zip', 'application/zip')
        if kind == 'bands.csv':
            return binary(csv_bytes(reports.band_table(cube)), 'band-statistics.csv', 'text/csv')
        if kind == 'objects.csv':
            return binary(csv_bytes(reports.object_table(asset)), 'object-measurements.csv', 'text/csv')
        if kind == 'regions.csv':
            selected = [row for row in project['regions'] if row['asset_id'] == asset_id]
            return binary(csv_bytes(regions.aggregate_region_table(cube, selected)), 'region-statistics.csv', 'text/csv')
        if kind == 'pixels.csv':
            return binary(csv_bytes(reports.pixel_table(cube, {'stride': request.args.get('stride', 8)})), 'sampled-pixels.csv', 'text/csv')
        if kind == 'geometry':
            data = geometry.label_geometry(cube, {})
            return binary(json.dumps(data, ensure_ascii=False, indent=2).encode(), 'object-boundaries.json', 'application/json')
        if kind == 'manifest':
            return response(reports.asset_manifest(asset, cube))
        raise c.Invalid('未知导出格式')

    @app.get('/api/projects/<project_id>/jobs')
    def project_jobs(project_id):
        store.get(project_id)
        return response({'jobs': jobs.list(project_id)})

    @app.get('/api/jobs/<job_id>')
    def job(job_id):
        return response({'job': jobs.get(job_id)})

    @app.post('/api/jobs/<job_id>/cancel')
    def cancel_job(job_id):
        return response({'job': jobs.cancel(job_id)})

    @app.delete('/api/projects/<project_id>/jobs')
    def clear_jobs(project_id):
        store.get(project_id)
        return response(jobs.purge_finished(project_id))

    @app.get('/api/projects/<project_id>/views')
    def list_views(project_id):
        store.get(project_id)
        return response(views.list(project_id))

    @app.post('/api/projects/<project_id>/views')
    def save_view(project_id):
        data = payload()
        _, _, cube = load_asset(project_id, data.get('asset_id'))
        return response(views.save(project_id, data.get('view_revision'), cube, data.get('name'), data.get('view', {})))

    @app.post('/api/projects/<project_id>/views/<preset_id>/resolve')
    def resolve_view(project_id, preset_id):
        data = payload()
        _, _, cube = load_asset(project_id, data.get('asset_id'))
        return response(views.resolve(project_id, preset_id, cube))

    @app.delete('/api/projects/<project_id>/views/<preset_id>')
    def remove_view(project_id, preset_id):
        data = payload()
        return response(views.remove(project_id, data.get('view_revision'), preset_id))

    @app.get('/api/projects/<project_id>/views/export')
    def export_views(project_id):
        store.get(project_id)
        return response(views.export(project_id))

    @app.post('/api/projects/<project_id>/views/import')
    def import_views(project_id):
        store.get(project_id)
        data = payload()
        return response(views.import_presets(project_id, data.get('view_revision'), data.get('presets')))

    return app
