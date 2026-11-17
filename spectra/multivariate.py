import numpy as np
from scipy import ndimage

from . import contracts as c
from .model import derived_cube
from .numerics import stable_covariance, stable_eigen, chunk_ranges, deterministic_indices


def selected_bands(source, options):
    return c.index_list(options.get('bands', list(range(source.count))), source.count)


def principal_components(source, options, references):
    bands = selected_bands(source, options)
    if len(bands) < 2:
        raise c.Invalid('主成分分析至少选择2个波段')
    components = c.integer(options.get('components', min(3, len(bands))), '分量数', 1, len(bands))
    standardize = c.boolean(options.get('standardize', False), '标准化')
    samples, indices = source.samples(bands, 50000)
    mean = samples.mean(axis=0)
    scales = samples.std(axis=0) if standardize else np.ones(len(bands))
    usable_bands = scales > 1e-10
    if usable_bands.sum() < components:
        raise c.Invalid('非恒定波段数量少于所需分量')
    scales = np.where(usable_bands, scales, 1)
    centered = (samples - mean) / scales
    covariance = centered.T @ centered / max(len(samples) - 1, 1)
    eigenvalues, eigenvectors = stable_eigen(covariance)
    vectors = eigenvectors[:, :components]
    values = source.values[:, :, bands].astype(float)
    scores = ((values - mean) / scales) @ vectors
    valid = source.common_valid(bands)
    report = {
        'bands': bands,
        'components': components,
        'mean': mean.tolist(),
        'scales': scales.tolist(),
        'loadings': vectors.tolist(),
        'eigenvalues': eigenvalues.tolist(),
        'explained_fraction': (eigenvalues / max(eigenvalues.sum(), 1e-12)).tolist(),
        'fit_sample_count': len(indices),
        'available_count': int(valid.sum()),
        'standardize': standardize,
        'basis_policy': 'one deterministic complete-spectrum covariance basis for the whole image',
    }
    names = [f'PC{i + 1}' for i in range(components)]
    return derived_cube(source, scores, valid, names, 'pca', report, 'score'), report


def noise_adjusted_components(source, options, references):
    bands = selected_bands(source, options)
    if len(bands) < 2:
        raise c.Invalid('噪声调整分量至少2波段')
    components = c.integer(options.get('components', min(3, len(bands))), '分量数', 1, len(bands))
    regularization = c.number(options.get('regularization', 0.001), '正则比例', 1e-8, 1)
    values = source.values[:, :, bands].astype(float)
    complete = source.common_valid(bands)
    pair_valid = complete[:, 1:] & complete[:, :-1]
    noise = (values[:, 1:] - values[:, :-1]) / np.sqrt(2)
    selected = deterministic_indices(pair_valid, 50000)
    if len(selected) < len(bands) + 2:
        raise c.Invalid('共同有效相邻像素对不足')
    noise_samples = noise.reshape(-1, len(bands))[selected]
    noise_mean, noise_covariance = stable_covariance(noise_samples, regularization)
    noise_values, noise_vectors = stable_eigen(noise_covariance)
    whitening = noise_vectors @ np.diag(1 / np.sqrt(np.maximum(noise_values, 1e-12))) @ noise_vectors.T
    samples, indices = source.samples(bands, 50000)
    mean = samples.mean(axis=0)
    transformed = (samples - mean) @ whitening
    covariance = transformed.T @ transformed / (len(samples) - 1)
    eigenvalues, eigenvectors = stable_eigen(covariance)
    loadings = whitening @ eigenvectors[:, :components]
    scores = (values - mean) @ loadings
    report = {
        'bands': bands,
        'components': components,
        'mean': mean.tolist(),
        'noise_covariance': noise_covariance.tolist(),
        'loadings': loadings.tolist(),
        'eigenvalues': eigenvalues.tolist(),
        'fit_samples': len(indices),
        'noise_pairs': len(selected),
        'noise_estimator': 'horizontal adjacent differences; material edges may contribute',
        'regularization': regularization,
    }
    names = [f'Noise-adjusted {i + 1}' for i in range(components)]
    return derived_cube(source, scores, complete, names, 'noise_components', report, 'score'), report


def initialize_centers(samples, count, rng):
    centers = [samples[rng.integers(len(samples))].copy()]
    minimum_distances = np.sum((samples - centers[0]) ** 2, axis=1)
    for _ in range(1, count):
        total = minimum_distances.sum()
        if total <= 1e-12:
            raise c.Invalid('有效光谱的不同取值少于所需聚类数')
        chosen = int(rng.choice(len(samples), p=minimum_distances / total))
        centers.append(samples[chosen].copy())
        distance = np.sum((samples - centers[-1]) ** 2, axis=1)
        minimum_distances = np.minimum(minimum_distances, distance)
    return np.asarray(centers)


def assign_clusters(samples, centers):
    distances = np.sum(samples ** 2, axis=1)[:, None] + np.sum(centers ** 2, axis=1)[None, :] - 2 * samples @ centers.T
    labels = np.argmin(distances, axis=1)
    errors = np.maximum(distances[np.arange(len(samples)), labels], 0)
    return labels, errors


def cluster_materials(source, options, references):
    bands = selected_bands(source, options)
    count = c.integer(options.get('clusters', 5), '聚类数', 2, 20)
    iterations = c.integer(options.get('iterations', 50), '最大迭代', 5, 200)
    standardize = c.boolean(options.get('standardize', True), '标准化')
    samples, selected = source.samples(bands, 30000)
    if len(samples) < count * 3:
        raise c.Invalid('聚类样本不足')
    mean = samples.mean(axis=0)
    scale = samples.std(axis=0) if standardize else np.ones(len(bands))
    scale = np.maximum(scale, 1e-8)
    training = (samples - mean) / scale
    rng = np.random.default_rng(71)
    centers = initialize_centers(training, count, rng)
    convergence = []
    for iteration in range(iterations):
        labels, errors = assign_clusters(training, centers)
        updated = centers.copy()
        for label in range(count):
            members = training[labels == label]
            if len(members):
                updated[label] = members.mean(axis=0)
            else:
                index = int(np.argmax(errors))
                updated[label] = training[index]
                errors[index] = -1
        change = float(np.max(np.linalg.norm(updated - centers, axis=1)))
        centers = updated
        convergence.append({'iteration': iteration + 1, 'maximum_center_change': change, 'mean_squared_distance': float(np.maximum(errors, 0).mean())})
        if change < 1e-6:
            break
    physical_centers = centers * scale + mean
    ordering = np.argsort(physical_centers.mean(axis=1), kind='stable')
    centers = centers[ordering]
    physical_centers = physical_centers[ordering]
    valid = source.common_valid(bands)
    flat = source.values[:, :, bands].reshape(-1, len(bands))
    pixels = np.flatnonzero(valid)
    output = np.zeros(flat.shape[0], dtype=np.float32)
    residual = np.zeros(flat.shape[0], dtype=np.float32)
    for start, end in chunk_ranges(len(pixels)):
        indices = pixels[start:end]
        labels, errors = assign_clusters((flat[indices] - mean) / scale, centers)
        output[indices] = labels + 1
        residual[indices] = np.sqrt(errors)
    report = {
        'bands': bands,
        'clusters': count,
        'training_samples': len(selected),
        'standardize': standardize,
        'mean': mean.tolist(),
        'scales': scale.tolist(),
        'centers': physical_centers.tolist(),
        'counts': [int((output == i + 1).sum()) for i in range(count)],
        'convergence': convergence,
        'label_semantics': 'unsupervised groups ordered by center mean; not named material classes',
        'mean_assignment_distance': float(residual[pixels].mean()),
    }
    return derived_cube(source, output.reshape(source.shape[:2]), valid, ['Spectral cluster'], 'cluster', report, 'cluster_id', domain='labels'), report


def anomaly_detection(source, options, references):
    bands = selected_bands(source, options)
    regularization = c.number(options.get('regularization', 0.001), '协方差正则', 1e-8, 1)
    samples, selected = source.samples(bands, 50000)
    center, covariance = stable_covariance(samples, regularization)
    inverse = np.linalg.pinv(covariance, hermitian=True)
    valid = source.common_valid(bands)
    flat = source.values[:, :, bands].reshape(-1, len(bands))
    indices = np.flatnonzero(valid)
    score = np.zeros(len(flat), dtype=float)
    for start, end in chunk_ranges(len(indices)):
        pixels = indices[start:end]
        difference = flat[pixels] - center
        score[pixels] = np.maximum(np.einsum('ni,ij,nj->n', difference, inverse, difference), 0)
    percentile = c.number(options.get('flag_percentile', 99), '标记分位', 50, 100)
    threshold = float(np.percentile(score[indices], percentile))
    report = {
        'bands': bands,
        'center': center.tolist(),
        'covariance': covariance.tolist(),
        'regularization': regularization,
        'fit_samples': len(selected),
        'threshold_percentile': percentile,
        'threshold': threshold,
        'above_threshold_pixels': int((score[indices] >= threshold).sum()),
        'model': 'global regularized squared Mahalanobis distance',
        'interpretation': 'unusual relative to this image; not probability of a defect',
    }
    return derived_cube(source, score.reshape(source.shape[:2]), valid, ['Spectral anomaly'], 'anomaly', report, 'squared_distance'), report


def reconstruct_components(source, options, references):
    if source.metadata.get('operation') != 'pca':
        raise c.Invalid('此操作需要由本工程PCA生成的影像')
    basis = source.metadata.get('report', {})
    loadings = np.asarray(basis.get('loadings'), dtype=float)
    mean = np.asarray(basis.get('mean'), dtype=float)
    scales = np.asarray(basis.get('scales'), dtype=float)
    if loadings.ndim != 2 or loadings.shape[1] != source.count:
        raise c.Invalid('PCA基底元数据不完整')
    components = c.integer(options.get('components', source.count), '保留分量数', 1, source.count)
    reconstructed = (source.values[:, :, :components] @ loadings[:, :components].T) * scales + mean
    valid = source.common_valid(list(range(components)))
    report = {'components': components, 'source_bands': basis['bands'], 'retained_variance_fraction': float(sum(basis['explained_fraction'][:components])), 'note': '低秩近似，保留为派生域'}
    names = [f'Reconstructed B{index}' for index in basis['bands']]
    return derived_cube(source, reconstructed, valid, names, 'pca_reconstruct', report, 'reconstructed_value'), report
