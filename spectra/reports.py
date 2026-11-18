import io
import json
import zipfile

import numpy as np
from PIL import Image

from . import contracts as c
from .io import csv_bytes, spectrum_csv, cube_bytes, band_tiff, mask_png
from .regions import aggregate_region_table, region_report
from .rendering import render_png, overlay_regions, band_contact_sheet, default_view
from . import diagnostics


def asset_manifest(asset, cube):
    return {
        'schema': 'materialscope_result_manifest_v1',
        'asset_id': asset['id'],
        'name': asset['name'],
        'created_at': asset['created_at'],
        'operation': asset['operation'],
        'parent_id': asset.get('parent_id'),
        'parameters': asset['parameters'],
        'report': asset['report'],
        'summary': cube.summary(),
        'fingerprint': cube.fingerprint(),
        'data_convention': {
            'array_order': 'height,width,band',
            'dtype': 'float32',
            'validity': 'per-band boolean; stored zero in invalid cells is not a measurement',
            'pixel_centers': 'origin + (index+0.5)*step',
            'wavelength_unit': 'nm',
            'display_stretch_changes_data': False,
        },
    }


def band_table(cube):
    rows = []
    for index, band in enumerate(cube.bands):
        quality = diagnostics.band_quality_report(cube, index)
        statistics = quality['statistics']
        rows.append({
            'band': index,
            'name': band['name'],
            'wavelength_nm': band['wavelength_nm'],
            'fwhm_nm': band['fwhm_nm'],
            'valid_samples': statistics['count'],
            'minimum': statistics['min'],
            'maximum': statistics['max'],
            'mean': statistics['mean'],
            'median': statistics['median'],
            'standard_deviation': statistics['std'],
            'unit': cube.unit,
            'domain': cube.domain,
        })
    return rows


def object_table(asset):
    objects = asset.get('report', {}).get('objects')
    if objects is None:
        raise c.Invalid('当前结果没有对象测量表')
    rows = []
    for item in objects:
        row = {
            'label': item['label'],
            'pixels': item['pixels'],
            'area': item['area'],
            'area_unit': item['area_unit'],
            'centroid_x': item['centroid_world'][0],
            'centroid_y': item['centroid_world'][1],
            'major_length': item['major_length'],
            'minor_length': item['minor_length'],
            'orientation_degrees': item['orientation_degrees'],
            'pixel_edge_perimeter': item['pixel_edge_perimeter'],
            'circularity': item['circularity'],
            'touches_image_border': item['touches_image_border'],
            'touches_missing': item['touches_missing'],
        }
        if 'intensity' in item:
            row['intensity_mean'] = item['intensity']['mean']
            row['intensity_std'] = item['intensity']['std']
        rows.append(row)
    return rows


def signature_table(signatures):
    rows = []
    for signature in signatures:
        for index, (wave, value) in enumerate(zip(signature['wavelengths_nm'], signature['values'])):
            rows.append({
                'signature_id': signature['id'],
                'material_name': signature['name'],
                'wavelength_nm': wave,
                'value': value,
                'standard_deviation': signature['std'][index] if signature.get('std') else None,
                'source_count': signature['source_counts'][index] if signature.get('source_counts') else None,
                'unit': signature['unit'],
                'domain': signature['domain'],
                'source_region': signature.get('source_region'),
            })
    return rows


def report_bundle(project, asset, cube, options):
    selected_regions = [region for region in project['regions'] if region['asset_id'] == asset['id']]
    options = c.object_value(options, '报告选项')
    view = options.get('view', default_view(cube))
    include_bands = c.boolean(options.get('include_band_tiffs', False), '逐带TIFF')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json', json.dumps(asset_manifest(asset, cube), ensure_ascii=False, indent=2))
        archive.writestr('cube.npz', cube_bytes(cube))
        archive.writestr('preview.png', render_png(cube, view))
        archive.writestr('band-contact-sheet.png', band_contact_sheet(cube, {'columns': 4}))
        archive.writestr('common-valid.png', mask_png(cube))
        archive.writestr('band-statistics.csv', csv_bytes(band_table(cube)))
        archive.writestr('validity.json', json.dumps(diagnostics.validity_report(cube), ensure_ascii=False))
        archive.writestr('spectral-grid.json', json.dumps(diagnostics.spectral_spacing_report(cube), ensure_ascii=False))
        if selected_regions:
            archive.writestr('regions-preview.png', overlay_regions(cube, view, selected_regions))
            archive.writestr('region-statistics.csv', csv_bytes(aggregate_region_table(cube, selected_regions)))
            archive.writestr('regions.json', json.dumps([region_report(cube, region) for region in selected_regions], ensure_ascii=False))
        if asset.get('report', {}).get('objects') is not None:
            archive.writestr('object-measurements.csv', csv_bytes(object_table(asset)))
        if project['signatures']:
            archive.writestr('material-library.csv', csv_bytes(signature_table(project['signatures'])))
        if include_bands:
            for index in range(cube.count):
                archive.writestr(f'bands/{index:02d}.tiff', band_tiff(cube, index))
                archive.writestr(f'masks/{index:02d}.png', mask_png(cube, index))
        archive.writestr('READ-ME.txt', 'MaterialScope local measurement report. Preview is display stretched; numerical values are in cube.npz and optional float TIFFs. Per-band valid masks are authoritative. Synthetic demo remains synthetic. No identification or chemical composition guarantee is implied. Pixel-edge areas and local coordinates are used; no geospatial projection is embedded.\n')
    if output.tell() > c.LIMITS['archive_bytes']:
        raise c.Invalid('报告包超过120MB')
    return output.getvalue()


def change_decision_rows(record, decision_cube):
    """变化判定逐像素表，与已固化方案的标签完全一致。"""
    labels = decision_cube.values[:, :, 0]
    height, width = labels.shape
    legend = decision_cube.metadata.get('label_legend', {})
    policy = record.get('policy')
    rows = []
    for y in range(height):
        for x in range(width):
            code = int(labels[y, x])
            rows.append({
                'x': x,
                'y': y,
                'decision_label': code,
                'decision_meaning': legend.get(str(code), str(code)),
                'policy': policy,
                'change_analysis_id': record['id'],
            })
    return rows


def project_inventory(project):
    rows = []
    for asset in project['assets']:
        summary = asset['summary']
        rows.append({
            'asset_id': asset['id'],
            'name': asset['name'],
            'operation': asset['operation'],
            'parent_id': asset.get('parent_id'),
            'width': summary['width'],
            'height': summary['height'],
            'bands': summary['band_count'],
            'domain': summary['domain'],
            'unit': summary['unit'],
            'common_valid_pixels': summary['common_valid_count'],
            'created_at': asset['created_at'],
            'digest': asset['digest'],
        })
    return rows


def pixel_table(cube, options):
    band_indices = c.index_list(options.get('bands', list(range(cube.count))), cube.count)
    stride = c.integer(options.get('stride', 8), '采样间距', 1, 128)
    maximum = c.integer(options.get('maximum_rows', 50000), '最大行数', 1, 100000)
    coordinates = [(x, y) for y in range(0, cube.height, stride) for x in range(0, cube.width, stride)]
    if len(coordinates) > maximum:
        raise c.Invalid('采样表超过行数预算，请增加间距')
    rows = []
    for x, y in coordinates:
        world = cube.pixel_center(x, y)
        row = {'x': x, 'y': y, 'world_x': world[0], 'world_y': world[1]}
        for band in band_indices:
            valid = bool(cube.valid[y, x, band])
            row[f'b{band}_value'] = float(cube.values[y, x, band]) if valid else None
            row[f'b{band}_valid'] = valid
        rows.append(row)
    return rows


def export_band_stack(cube):
    output = io.BytesIO()
    manifest = {
        'schema': 'materialscope_band_stack_v1',
        'domain': cube.domain,
        'unit': cube.unit,
        'grid': cube.grid,
        'bands': [],
    }
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for index, band in enumerate(cube.bands):
            filename = f'band-{index:02d}.tiff'
            archive.writestr(filename, band_tiff(cube, index))
            manifest['bands'].append({
                'file': filename,
                'channel': 0,
                'name': band['name'],
                'wavelength_nm': band['wavelength_nm'],
                'fwhm_nm': band['fwhm_nm'],
                'gain': 1,
                'offset': 0,
            })
        archive.writestr('bands.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    return output.getvalue()


def export_raw(cube, options):
    layout = c.choice(options.get('interleave', 'BIL'), '排列', ['BIL', 'BIP', 'BSQ'])
    endian = c.choice(options.get('byte_order', 'little'), '字节序', ['little', 'big'])
    values = np.where(cube.valid, cube.values, np.nan)
    if layout == 'BIL':
        arranged = values.transpose(0, 2, 1)
    elif layout == 'BSQ':
        arranged = values.transpose(2, 0, 1)
    else:
        arranged = values
    dtype = '<f4' if endian == 'little' else '>f4'
    data = np.asarray(arranged, dtype=dtype).tobytes()
    specification = {
        'width': cube.width,
        'height': cube.height,
        'band_count': cube.count,
        'dtype': 'float32',
        'byte_order': endian,
        'interleave': layout,
        'header_bytes': 0,
        'row_padding_bytes': 0,
        'bands': list(cube.bands),
        'grid': cube.grid,
        'unit': cube.unit,
        'domain': cube.domain,
        'invalid_encoding': 'IEEE NaN; raw import preserves validity but imports into raw domain',
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('capture.raw', data)
        archive.writestr('capture.json', json.dumps(specification, ensure_ascii=False, indent=2))
    return output.getvalue()
