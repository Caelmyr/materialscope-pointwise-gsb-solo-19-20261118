import copy
import hashlib
import json

import numpy as np

from . import contracts as c
from .model import derived_cube
from .regions import region_mask


ORIENTATIONS = ['foreign_sensitive', 'known_conservative']
ORIENTATION_LABELS = {
    'pending': '未决',
    'foreign_sensitive': '少漏异物优先',
    'known_conservative': '少误报优先',
}
CONFIDENCE_CAP_MISSING = 0.69


def collect_labeled(cube, regions):
    classes = {}
    negatives = []
    for row in regions:
        training = row.get('training') or {}
        role = training.get('role')
        if role not in ('class', 'negative'):
            continue
        mask = region_mask(cube, row)
        entry = {'region': row, 'mask': mask}
        if role == 'class':
            classes.setdefault(training['name'], []).append(entry)
        else:
            negatives.append(entry)
    if not classes:
        raise c.Invalid('当前影像没有已标记的材料分区')
    assignments = []
    for name, entries in classes.items():
        for entry in entries:
            assignments.append((entry['mask'], '材料「' + name + '」'))
    for entry in negatives:
        assignments.append((entry['mask'], '反例'))
    unique_labels = []
    for _, label in assignments:
        if label not in unique_labels:
            unique_labels.append(label)
    owner = np.zeros(cube.shape[:2], dtype=np.int32)
    for mask, label in assignments:
        index = unique_labels.index(label) + 1
        conflict = mask & (owner != 0) & (owner != index)
        if conflict.any():
            raise c.Invalid(f'分区重叠处被同时标记为不同类别（{label}，{int(conflict.sum())}像素），请调整分区边界')
        owner[mask] = index
    return classes, negatives


def extract_samples(cube, mask, bands, maximum, seed=71):
    complete = mask & cube.common_valid(bands)
    indices = np.flatnonzero(complete)
    total = len(indices)
    if total > maximum:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, maximum, replace=False))
    values = cube.values[:, :, bands].reshape(-1, len(bands))[indices].astype(np.float64)
    return values, total


def fit_gaussian(class_samples, shrinkage):
    centers = np.stack([samples.mean(axis=0) for samples in class_samples])
    count = centers.shape[1]
    covariances = []
    for samples in class_samples:
        if len(samples) > count + 1:
            centered = samples - samples.mean(axis=0)
            covariances.append(centered.T @ centered / (len(samples) - 1))
        else:
            variances = samples.var(axis=0) if len(samples) > 1 else np.zeros(count)
            covariances.append(np.diag(variances + 1e-8))
    pooled = np.mean(covariances, axis=0)
    pooled = (1 - shrinkage) * pooled + shrinkage * np.diag(np.diag(pooled))
    floor = max(float(np.trace(pooled) / count), 1e-12) * 1e-6
    return centers, pooled + np.eye(count) * floor


def fisher_key_bands(centers, covariance, coverage):
    if len(centers) < 2:
        return [], []
    between = centers.var(axis=0)
    within = np.diag(covariance)
    importance = between / np.maximum(within, 1e-12)
    total = float(importance.sum())
    if total <= 1e-12:
        return [], [0.0] * centers.shape[1]
    order = np.argsort(importance)[::-1]
    cumulative = np.cumsum(importance[order]) / total
    count = int(np.searchsorted(cumulative, coverage) + 1)
    count = max(1, min(count, len(order)))
    return sorted(order[:count].tolist()), importance.tolist()


def _compiled(model):
    return {
        'centers': np.asarray(model['centers'], dtype=np.float64),
        'covariance': np.asarray(model['covariance'], dtype=np.float64),
        'key_bands': np.asarray(model['key_bands'], dtype=np.int64),
        'min_bands': int(model['min_bands']),
    }


def _distances_for_pattern(compiled, samples, selected):
    centers = compiled['centers'][:, selected]
    covariance = compiled['covariance'][np.ix_(selected, selected)]
    inverse = np.linalg.inv(covariance)
    output = np.empty((len(samples), centers.shape[0]))
    for k in range(centers.shape[0]):
        for start in range(0, len(samples), 100000):
            end = min(start + 100000, len(samples))
            diff = samples[start:end] - centers[k]
            output[start:end, k] = np.maximum(np.einsum('ni,ij,nj->n', diff, inverse, diff), 0)
    return output


def predict_patterns(model, thresholds, values, observed):
    compiled = _compiled(model)
    centers = compiled['centers']
    class_count, band_count = centers.shape
    accept = float(thresholds['accept_distance'])
    margin = float(thresholds['margin_loglikelihood'])
    n = len(values)
    codes = np.zeros(n, dtype=np.int64)
    confidence = np.zeros(n, dtype=np.float64)
    best_delta = np.full(n, np.nan)
    if n == 0:
        return codes, confidence, best_delta
    unknown_code = class_count + 1
    insufficient_code = class_count + 2
    weights = np.left_shift(np.ones(band_count, dtype=np.uint64), np.arange(band_count, dtype=np.uint64))
    keys = observed.astype(np.uint64) @ weights
    unique, inverse = np.unique(keys, return_inverse=True)
    key_bands = compiled['key_bands']
    for position in range(len(unique)):
        indices = np.flatnonzero(inverse == position)
        key = unique[position]
        pattern = np.array([(key >> np.uint64(b)) & np.uint64(1) for b in range(band_count)], dtype=bool)
        count = int(pattern.sum())
        if count < compiled['min_bands']:
            codes[indices] = insufficient_code
            continue
        selected = np.flatnonzero(pattern)
        distances = _distances_for_pattern(compiled, values[indices][:, selected], selected)
        order = np.argsort(distances, axis=1, kind='stable')
        best = order[:, 0]
        rows = np.arange(len(indices))
        d_best = distances[rows, best]
        delta_best = d_best / count
        d_second = distances[rows, order[:, 1]] if class_count > 1 else np.full(len(indices), np.inf)
        missing_key = bool(key_bands.size) and bool((~pattern[key_bands]).any())
        unknown = delta_best > accept
        if missing_key and class_count > 1:
            ambiguous = (d_second - d_best) < margin
        else:
            ambiguous = np.zeros(len(indices), dtype=bool)
        labeled = ~unknown & ~ambiguous
        codes[indices] = np.where(unknown, unknown_code, np.where(ambiguous, insufficient_code, best + 1))
        accept_ratio = np.clip(1 - delta_best / max(accept, 1e-12), 0, 1)
        margin_ratio = np.clip((d_second - d_best) / margin, 0, 1) if margin > 0 else np.ones(len(indices))
        conf = 0.5 * accept_ratio + 0.5 * margin_ratio
        if count < band_count:
            conf = np.minimum(conf, CONFIDENCE_CAP_MISSING)
        confidence[indices] = np.where(labeled, conf, 0.0)
        best_delta[indices] = delta_best
    return codes, confidence, best_delta


def train_classifier(cube, asset_id, asset_name, labeled_regions, name, options):
    options = c.object_value(options, '训练参数')
    if cube.domain != 'reflectance' or cube.unit != 'fraction':
        raise c.Invalid('分类器训练需要fraction反射率影像')
    bands = c.index_list(options.get('bands', list(range(cube.count))), cube.count, '波段', 3)
    waves = cube.wavelengths()[bands]
    band_count = len(bands)
    min_bands = c.integer(options.get('min_bands', 3), '最少有效波段', 1, band_count)
    shrinkage = c.number(options.get('shrinkage', 0.15), '协方差收缩', 0, 0.9)
    margin = c.number(options.get('margin_loglikelihood', 2.0), '歧义对数似然余量', 0, 50)
    min_region_pixels = c.integer(options.get('min_region_pixels', 5), '区域最少完整像素', 1, 1000000)
    min_class_pixels = c.integer(options.get('min_class_pixels', 30), '类最少完整像素', 2, 16000000)
    min_negative_pixels = c.integer(options.get('min_negative_pixels', 20), '反例最少完整像素', 0, 16000000)
    key_coverage = c.number(options.get('key_band_coverage', 0.5), '关键波段覆盖', 0.1, 0.95)
    max_train = c.integer(options.get('max_train_pixels_per_class', 20000), '类训练像素上限', 100, 1000000)
    max_validation = c.integer(options.get('max_validation_pixels_per_region', 50000), '区域验证像素上限', 10, 1000000)
    classes_map, negatives = collect_labeled(cube, labeled_regions)
    common = cube.common_valid(bands)
    class_names = sorted(classes_map)
    class_ids = [c.generated_id('class') for _ in class_names]
    reasons = []
    class_entries = []
    for position, class_name in enumerate(class_names):
        entries = classes_map[class_name]
        mask = np.zeros(cube.shape[:2], dtype=bool)
        for entry in entries:
            mask |= entry['mask']
        samples, total = extract_samples(cube, mask, bands, max_train)
        if total < 2:
            raise c.Invalid(f'材料「{class_name}」完整光谱像素不足2个，无法训练')
        class_entries.append({
            'name': class_name,
            'id': class_ids[position],
            'regions': entries,
            'mask': mask,
            'samples': samples,
            'pixels': total,
        })
        if total < min_class_pixels:
            reasons.append(f'材料「{class_name}」完整光谱像素{total}少于{min_class_pixels}')
        if len(entries) < 2:
            reasons.append(f'材料「{class_name}」只有{len(entries)}块区域，整块留出验证需要每类至少2块独立区域')
        for entry in entries:
            complete = int((entry['mask'] & common).sum())
            if complete < min_region_pixels:
                reasons.append(f'区域「{entry["region"]["name"]}」完整光谱像素{complete}少于{min_region_pixels}')
    negative_mask = np.zeros(cube.shape[:2], dtype=bool)
    for entry in negatives:
        negative_mask |= entry['mask']
        complete = int((entry['mask'] & common).sum())
        if complete < min_region_pixels:
            reasons.append(f'反例区域「{entry["region"]["name"]}」完整光谱像素{complete}少于{min_region_pixels}')
    neg_samples, neg_total = extract_samples(cube, negative_mask, bands, max_train) if negatives else (np.zeros((0, band_count)), 0)
    if not negatives:
        reasons.append('没有反例区域，无法估计异物漏检，误判取向保持未决')
    elif neg_total < min_negative_pixels:
        reasons.append(f'反例完整光谱像素{neg_total}少于{min_negative_pixels}')
    centers, covariance = fit_gaussian([entry['samples'] for entry in class_entries], shrinkage)
    key_bands, importance = fisher_key_bands(centers, covariance, key_coverage)
    region_units = []
    for entry in class_entries:
        for region_entry in entry['regions']:
            region_units.append({'region': region_entry['region'], 'mask': region_entry['mask'], 'role': 'class', 'class_name': entry['name']})
    for entry in negatives:
        region_units.append({'region': entry['region'], 'mask': entry['mask'], 'role': 'negative', 'class_name': None})
    folds = []
    oof_true = []
    oof_best = []
    oof_delta_best = []
    oof_delta_true = []
    full_bands = np.arange(band_count)
    for unit in region_units:
        fold_samples = []
        fold_names = []
        for entry in class_entries:
            if unit['role'] == 'class' and unit['class_name'] == entry['name']:
                others = [item for item in entry['regions'] if item['region']['id'] != unit['region']['id']]
                if not others:
                    continue
                mask = np.zeros(cube.shape[:2], dtype=bool)
                for item in others:
                    mask |= item['mask']
            else:
                mask = entry['mask']
            samples, total = extract_samples(cube, mask, bands, max_train)
            if total < 2:
                continue
            fold_samples.append(samples)
            fold_names.append(entry['name'])
        fold = {
            'held_out_region_id': unit['region']['id'],
            'held_out_name': unit['region']['name'],
            'role': unit['role'],
            'class_name': unit['class_name'],
            'fold_classes': fold_names,
            'pixels': 0,
            'skipped_partial_pixels': int((unit['mask'] & ~common).sum()),
        }
        if not fold_names:
            fold['note'] = '留出后没有可训练类别，该区域无法验证'
            folds.append(fold)
            continue
        if unit['role'] == 'class' and unit['class_name'] not in fold_names:
            fold['note'] = '该类别在留出后没有其他区域，无法验证'
            folds.append(fold)
            continue
        fold_centers, fold_covariance = fit_gaussian(fold_samples, shrinkage)
        compiled = {'centers': fold_centers, 'covariance': fold_covariance, 'key_bands': np.zeros(0, dtype=np.int64), 'min_bands': 1}
        samples, total = extract_samples(cube, unit['mask'], bands, max_validation)
        fold['pixels'] = int(total)
        folds.append(fold)
        if total == 0:
            continue
        distances = _distances_for_pattern(compiled, samples, full_bands)
        order = np.argsort(distances, axis=1, kind='stable')
        best = order[:, 0]
        rows = np.arange(len(samples))
        d_best = distances[rows, best]
        if unit['role'] == 'class':
            true_global = class_names.index(unit['class_name'])
            d_true = distances[:, fold_names.index(unit['class_name'])]
            oof_true.extend([true_global] * len(samples))
            oof_delta_true.extend((d_true / band_count).tolist())
        else:
            oof_true.extend([-1] * len(samples))
            oof_delta_true.extend([np.nan] * len(samples))
        oof_best.extend([class_names.index(fold_names[index]) for index in best])
        oof_delta_best.extend((d_best / band_count).tolist())
    oof_true = np.asarray(oof_true)
    oof_best = np.asarray(oof_best)
    oof_delta_best = np.asarray(oof_delta_best, dtype=np.float64)
    oof_delta_true = np.asarray(oof_delta_true, dtype=np.float64)
    known = oof_true >= 0
    negative = ~known
    delta_true_known = oof_delta_true[known]
    delta_negative = oof_delta_best[negative]
    # 少漏异物：能接受的最紧阈值——已知材料误拒约5%，在此约束下尽量抓住异物
    t_sensitive = max(float(np.quantile(delta_true_known, 0.95)), 1e-9) if len(delta_true_known) else None
    # 少误报：能接受的最松阈值——反例漏检约5%，在此约束下尽量少误报正常材料
    t_conservative = max(float(np.quantile(delta_negative, 0.05)), 1e-9) if len(delta_negative) else None

    def metrics_for(threshold):
        accepted = oof_delta_best <= threshold
        per_class = []
        for k, class_name in enumerate(class_names):
            selected = known & (oof_true == k)
            pixels = int(selected.sum())
            correct = int((selected & accepted & (oof_best == k)).sum())
            wrong = int((selected & accepted & (oof_best != k)).sum())
            unknown = int((selected & ~accepted).sum())
            confusion = {}
            for j, other in enumerate(class_names):
                if j == k:
                    continue
                count = int((selected & accepted & (oof_best == j)).sum())
                if count:
                    confusion[other] = count
            per_class.append({
                'class_id': class_ids[k],
                'name': class_name,
                'pixels': pixels,
                'correct': correct,
                'wrong_class': wrong,
                'unknown': unknown,
                'insufficient': 0,
                'confused_with': confusion,
                'error_rate': wrong / pixels if pixels else None,
                'rejection_rate': unknown / pixels if pixels else None,
            })
        negative_pixels = int(negative.sum())
        negative_accepted = int((negative & accepted).sum())
        return {
            'per_class': per_class,
            'negative': {
                'pixels': negative_pixels,
                'accepted_as_known': negative_accepted,
                'rejected': negative_pixels - negative_accepted,
                'miss_rate': negative_accepted / negative_pixels if negative_pixels else None,
            },
            'totals': {
                'known_pixels': int(known.sum()),
                'correct': int((known & accepted & (oof_best == oof_true)).sum()),
                'wrong_class': int((known & accepted & (oof_best != oof_true)).sum()),
                'unknown': int((known & ~accepted).sum()),
                'negative_pixels': negative_pixels,
                'negative_accepted': negative_accepted,
            },
        }

    orientations = {}
    if t_sensitive is not None:
        orientations['foreign_sensitive'] = {
            'available': True,
            'intent': '少漏掉异物：接受阈值收紧到已知材料真类距离的95%分位，异物几乎都被拒为未知，代价是约5%已知材料被误拒',
            'thresholds': {'accept_distance': t_sensitive, 'margin_loglikelihood': margin, 'min_bands': min_bands},
            'metrics': metrics_for(t_sensitive),
        }
    else:
        orientations['foreign_sensitive'] = {'available': False, 'reason': '没有可整块留出的已知材料像素'}
    if t_conservative is not None:
        orientations['known_conservative'] = {
            'available': True,
            'intent': '少误报正常材料：接受阈值放宽到反例最近距离的5%分位，已知材料几乎不被误拒，代价是约5%与反例相似的异物可能被接受',
            'thresholds': {'accept_distance': t_conservative, 'margin_loglikelihood': margin, 'min_bands': min_bands},
            'metrics': metrics_for(t_conservative),
        }
    else:
        orientations['known_conservative'] = {'available': False, 'reason': '没有反例像素，无法估计异物接受阈值'}
    warnings = []
    if t_sensitive is not None and t_conservative is not None and t_sensitive > t_conservative:
        warnings.append('已知材料与反例的距离分布重叠，两种取向的阈值松紧关系倒置，请仔细比较验证表现再选择')
    training_regions = []
    for unit in region_units:
        region = unit['region']
        training_regions.append({
            'id': region['id'],
            'name': region['name'],
            'kind': region['kind'],
            'bounds': list(region['bounds']),
            'role': unit['role'],
            'class_name': unit['class_name'],
        })
    row = {
        'id': c.generated_id('classifier'),
        'name': c.text(name or '材料分类器', '分类器名称'),
        'created_at': c.timestamp(),
        'source_asset_id': c.identifier(asset_id),
        'source_asset_name': c.text(asset_name, '来源影像名称'),
        'source_fingerprint': cube.fingerprint(),
        'domain': cube.domain,
        'unit': cube.unit,
        'band_indices': list(bands),
        'wavelengths_nm': [float(wave) for wave in waves],
        'classes': [
            {
                'id': entry['id'],
                'name': entry['name'],
                'region_ids': [item['region']['id'] for item in entry['regions']],
                'pixels': int(entry['pixels']),
                'training_pixels': int(len(entry['samples'])),
            }
            for entry in class_entries
        ],
        'negative': {
            'region_ids': [entry['region']['id'] for entry in negatives],
            'pixels': int(neg_total),
            'training_pixels': int(len(neg_samples)),
        },
        'training_regions': training_regions,
        'options': {
            'bands': list(bands),
            'min_bands': min_bands,
            'shrinkage': shrinkage,
            'margin_loglikelihood': margin,
            'min_region_pixels': min_region_pixels,
            'min_class_pixels': min_class_pixels,
            'min_negative_pixels': min_negative_pixels,
            'key_band_coverage': key_coverage,
            'max_train_pixels_per_class': max_train,
            'max_validation_pixels_per_region': max_validation,
        },
        'model': {
            'method': 'gaussian_shared_covariance',
            'centers': centers.tolist(),
            'covariance': covariance.tolist(),
            'key_bands': key_bands,
            'band_importance': importance,
            'min_bands': min_bands,
            'class_weighting': 'equal per class; region size and class count imbalance do not change class weights',
        },
        'validation': {
            'scheme': 'leave_one_region_out',
            'sufficient': not reasons,
            'insufficiency_reasons': reasons,
            'warnings': warnings,
            'folds': folds,
            'orientations': orientations,
            'out_of_fold_pixels': int(len(oof_true)),
            'note': '阈值由同一份整块区域留出预测校准；每类错误、拒识、覆盖与反例漏检分开报告，不用单一总体准确率代替',
        },
        'orientation': 'pending',
        'decision': None,
    }
    return row


def choose_orientation(row, orientation):
    orientation = c.choice(orientation, '误判取向', ['pending'] + ORIENTATIONS)
    row = copy.deepcopy(row)
    if orientation == 'pending':
        row['orientation'] = 'pending'
        row['decision'] = None
        return row
    if not row['validation'].get('sufficient'):
        raise c.Invalid('验证样本不足，不能选择取向：' + '；'.join(row['validation'].get('insufficiency_reasons', [])))
    info = row['validation']['orientations'].get(orientation)
    if not info or not info.get('available'):
        raise c.Invalid('该取向缺少验证估计，保持未决')
    row['orientation'] = orientation
    row['decision'] = {
        'orientation': orientation,
        'chosen_at': c.timestamp(),
        'thresholds': info['thresholds'],
    }
    return row


def classifier_fingerprint(row):
    return hashlib.sha256(json.dumps(c.json_safe(row), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def match_bands(cube, wavelengths, tolerance):
    cube_waves = cube.wavelengths()
    mapping = []
    used = set()
    for wave in wavelengths:
        differences = np.abs(cube_waves - wave)
        nearest = int(np.argmin(differences))
        if differences[nearest] <= tolerance and nearest not in used:
            mapping.append(nearest)
            used.add(nearest)
        else:
            mapping.append(None)
    return mapping


def decision_code_names(row):
    class_count = len(row['classes'])
    names = {str(index + 1): item['name'] for index, item in enumerate(row['classes'])}
    names[str(class_count + 1)] = '未知（明确不像已知材料）'
    names[str(class_count + 2)] = '证据不足（观测波段太少或关键区分波段缺失）'
    names['0'] = '无数据/未评估'
    return names


def apply_classifier(source, options, references):
    row = references.get('classifier')
    if row is None:
        raise c.Invalid('缺少已保存分类器')
    if row.get('orientation') == 'pending' or not row.get('decision'):
        raise c.Invalid('分类器误判取向未决，不能上线应用')
    if source.domain != 'reflectance' or source.unit != 'fraction':
        raise c.Invalid('保存分类器判定需要fraction反射率域')
    model = row['model']
    wavelengths = [float(wave) for wave in row['wavelengths_nm']]
    tolerance = c.number(options.get('wavelength_tolerance_nm', 1.0), '波长匹配容差', 0, 100)
    mapping = match_bands(source, wavelengths, tolerance)
    matched = [index for index, target in enumerate(mapping) if target is not None]
    if len(matched) < model['min_bands']:
        raise c.Invalid(f'可匹配波长不足：{len(matched)}/{model["min_bands"]}，不能判定')
    key_bands = set(model['key_bands'])
    missing = [index for index, target in enumerate(mapping) if target is None]
    missing_key = sorted(key_bands & set(missing))
    band_count = len(wavelengths)
    total = source.width * source.height
    values = np.zeros((total, band_count))
    observed = np.zeros((total, band_count), dtype=bool)
    for model_band, cube_band in enumerate(mapping):
        if cube_band is None:
            continue
        observed[:, model_band] = source.valid[:, :, cube_band].ravel()
        values[:, model_band] = source.values[:, :, cube_band].ravel()
    thresholds = row['decision']['thresholds']
    any_observed = observed.any(axis=1)
    codes = np.zeros(total, dtype=np.int64)
    confidence = np.zeros(total, dtype=np.float64)
    best_delta = np.full(total, np.nan)
    selected = np.flatnonzero(any_observed)
    codes[selected], confidence[selected], best_delta[selected] = predict_patterns(model, thresholds, values[selected], observed[selected])
    class_count = len(row['classes'])
    unknown_code = class_count + 1
    insufficient_code = class_count + 2
    shape = source.shape[:2]
    decision_valid = any_observed.reshape(shape)
    confidence_valid = ((codes >= 1) & (codes <= class_count)).reshape(shape)
    distance_valid = np.isfinite(best_delta).reshape(shape)
    output = np.stack([
        codes.astype(np.float32),
        confidence.astype(np.float32),
        np.where(np.isfinite(best_delta), best_delta, 0).astype(np.float32),
    ], axis=1).reshape(shape[0], shape[1], 3)
    output_valid = np.stack([decision_valid, confidence_valid, distance_valid], axis=2)
    code_names = decision_code_names(row)
    report = {
        'classifier_id': row['id'],
        'classifier_name': row['name'],
        'classifier_fingerprint': classifier_fingerprint(row),
        'on_training_image': bool(source.fingerprint() == row['source_fingerprint']),
        'orientation': row['orientation'],
        'orientation_label': ORIENTATION_LABELS[row['orientation']],
        'thresholds': thresholds,
        'band_mapping': [
            {'model_band': index, 'wavelength_nm': wavelengths[index], 'cube_band': mapping[index]}
            for index in matched
        ],
        'missing_model_bands': [
            {'model_band': index, 'wavelength_nm': wavelengths[index], 'key_band': index in key_bands}
            for index in missing
        ],
        'key_bands': sorted(key_bands),
        'key_bands_missing': missing_key,
        'min_bands': model['min_bands'],
        'wavelength_tolerance_nm': tolerance,
        'decision_codes': code_names,
        'outcomes': {
            'labeled_pixels': int(((codes >= 1) & (codes <= class_count)).sum()),
            'unknown_pixels': int((codes == unknown_code).sum()),
            'insufficient_pixels': int((codes == insufficient_code).sum()),
            'no_data_pixels': int((codes == 0).sum()),
        },
        'per_class': [
            {'class_id': item['id'], 'name': item['name'], 'pixels': int((codes == index + 1).sum())}
            for index, item in enumerate(row['classes'])
        ],
        'missing_band_policy': '按像素观测子集边缘化马氏距离，缺测波段从不补零；关键区分波段缺失且前两类距离差不足时判为证据不足；缺波段像素置信度封顶' + str(CONFIDENCE_CAP_MISSING),
        'decision_bands': ['Material decision', 'Decision confidence', 'Best distance'],
    }
    cube = derived_cube(
        source,
        output,
        output_valid,
        ['Material decision', 'Decision confidence', 'Best distance'],
        'apply_classifier',
        report,
        unit='class_id',
        domain='labels',
    )
    return cube, report


def explain_pixel(cube, row, x, y):
    if row.get('orientation') == 'pending' or not row.get('decision'):
        raise c.Invalid('分类器误判取向未决，没有可核对的判定')
    x = c.integer(x, '横坐标', 0, cube.width - 1)
    y = c.integer(y, '纵坐标', 0, cube.height - 1)
    model = row['model']
    wavelengths = [float(wave) for wave in row['wavelengths_nm']]
    mapping = match_bands(cube, wavelengths, 1.0)
    observed = np.array([[target is not None and bool(cube.valid[y, x, target]) for target in mapping]])
    values = np.array([[float(cube.values[y, x, target]) if target is not None else 0.0 for target in mapping]])
    class_names = [item['name'] for item in row['classes']]
    class_count = len(class_names)
    key_bands = set(model['key_bands'])
    missing = [index for index in range(len(wavelengths)) if not observed[0, index]]
    result = {
        'x': x,
        'y': y,
        'classifier_id': row['id'],
        'classifier_name': row['name'],
        'orientation': row['orientation'],
        'orientation_label': ORIENTATION_LABELS[row['orientation']],
        'thresholds': row['decision']['thresholds'],
        'min_bands': model['min_bands'],
        'observed_count': int(observed.sum()),
        'observed_bands': [
            {'model_band': index, 'wavelength_nm': wavelengths[index], 'cube_band': mapping[index], 'value': float(values[0, index])}
            for index in range(len(wavelengths)) if observed[0, index]
        ],
        'missing_bands': [
            {'model_band': index, 'wavelength_nm': wavelengths[index], 'key_band': index in key_bands}
            for index in missing
        ],
        'note': '判断依据按保存模型与1nm容差重算；缺测波段边缘化，不补零',
    }
    if not observed.any():
        result['outcome'] = {'code': 0, 'name': '无数据', 'reason': '该像素没有任何模型波段的有效观测'}
        return result
    thresholds = row['decision']['thresholds']
    codes, confidence, best_delta = predict_patterns(model, thresholds, values, observed)
    code = int(codes[0])
    selected = np.flatnonzero(observed[0])
    per_class = []
    if len(selected) >= model['min_bands']:
        compiled = _compiled(model)
        distances = _distances_for_pattern(compiled, values[:, selected], selected)[0]
        order = np.argsort(distances, kind='stable')
        for rank, k in enumerate(order):
            per_class.append({
                'name': class_names[k],
                'rank': rank + 1,
                'distance': float(distances[k]),
                'distance_normalized': float(distances[k] / len(selected)),
                'accepted': bool(distances[k] / len(selected) <= thresholds['accept_distance']),
            })
        result['margin'] = {
            'difference': float(distances[order[1]] - distances[order[0]]) if class_count > 1 else None,
            'required': thresholds['margin_loglikelihood'],
            'checked': bool(key_bands & set(missing)) and class_count > 1,
        }
    result['per_class'] = per_class
    result['confidence'] = float(confidence[0]) if 1 <= code <= class_count else None
    result['best_distance_normalized'] = float(best_delta[0]) if np.isfinite(best_delta[0]) else None
    if 1 <= code <= class_count:
        outcome = {'code': code, 'name': class_names[code - 1], 'reason': f'最近类距离在{ORIENTATION_LABELS[row["orientation"]]}阈值内'}
        if missing:
            outcome['reason'] += f'；缺{len(missing)}个波段但剩余波段足以区分，置信度已封顶'
    elif code == class_count + 1:
        outcome = {'code': code, 'name': '未知（明确不像已知材料）', 'reason': '最近类距离超过接受阈值'}
    else:
        if len(selected) < model['min_bands']:
            reason = f'有效波段{len(selected)}个少于最少要求{model["min_bands"]}个'
        else:
            reason = '关键区分波段缺失且前两类距离差小于歧义余量'
        outcome = {'code': code, 'name': '证据不足', 'reason': reason}
    result['outcome'] = outcome
    return result
