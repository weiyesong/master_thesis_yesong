"""Independent, complete-test NumPy validation of segmentation dropout-off.

No project model or metric routine is imported. Old deterministic and MC
artifacts are read directly to establish paired sample IDs, labels and masks.
The primary sums use float64 accumulation; foreground probability clipping and
log evaluation preserve the export's native float32 convention. Calibration
bins are (lower, upper], with zero assigned to the first bin. Empty bins remain
in the CSV. The script is incremental, and --require-complete requires all 12
protocol cells. It writes only dated completion verification products.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.special import softmax

ROOT = Path('/workspace')
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'reports/core_rq_audit_20260918'
ARTIFACT_ROOT = ROOT / 'results/final_thesis/core_rq_completion_20260921/segmentation'
TOLERANCE = 2e-6
VERSION = 2
KEYS = ('dataset', 'model', 'adaptation', 'seed')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True) + '\n')
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Bins:
    def __init__(self, count: int = 15):
        self.n = count
        self.count = np.zeros(count, dtype=np.int64)
        self.probability = np.zeros(count, dtype=np.float64)
        self.outcome = np.zeros(count, dtype=np.float64)

    def update(self, probability: np.ndarray, outcome: np.ndarray) -> None:
        p = np.asarray(probability, dtype=np.float64).reshape(-1)
        y = np.asarray(outcome).reshape(-1)
        indices = np.clip(np.searchsorted(np.linspace(0, 1, self.n + 1), p, side='left') - 1, 0, self.n - 1)
        self.count += np.bincount(indices, minlength=self.n)
        self.probability += np.bincount(indices, weights=p, minlength=self.n)
        self.outcome += np.bincount(indices, weights=y, minlength=self.n)

    def ece(self) -> float:
        total = self.count.sum()
        return float(np.abs(self.probability - self.outcome).sum() / total) if total else float('nan')

    def records(self, base: dict, kind: str) -> list[dict]:
        return [dict(base, kind=kind, n_bins=self.n, bin=i,
                     lower=i / self.n, upper=(i + 1) / self.n,
                     interval='[0, upper]' if i == 0 else '(lower, upper]',
                     count=int(self.count[i]),
                     mean_confidence=float(self.probability[i] / self.count[i]) if self.count[i] else '',
                     observed_frequency=float(self.outcome[i] / self.count[i]) if self.count[i] else '',
                     signed_confidence_minus_outcome=float((self.probability[i] - self.outcome[i]) / self.count[i]) if self.count[i] else '')
                for i in range(self.n)]


def key(row: dict) -> tuple:
    return tuple(str(row[k]) for k in KEYS)


def scalar_leaves(value: dict, prefix: str = '') -> dict:
    result = {}
    for name, item in value.items():
        name = prefix + name
        if isinstance(item, dict):
            result.update(scalar_leaves(item, name + '.'))
        elif isinstance(item, (float, int)):
            result[name] = item
    return result


def find_values(value: Any, name: str) -> list:
    if isinstance(value, dict):
        values = [value[name]] if name in value else []
        return values + [v for item in value.values() for v in find_values(item, name)]
    if isinstance(value, list):
        return [v for item in value for v in find_values(item, name)]
    return []


def source_pair_checks(source: dict, ids: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> dict:
    """Read old source arrays, rather than accepting the old audit conclusion."""
    path = ROOT / source['artifact_path']
    with np.load(path, allow_pickle=False) as archive:
        source_ids = archive['sample_id']
        source_labels = archive['label']
        source_valid = archive['valid_mask'].astype(bool)
    exact_order = np.array_equal(source_ids, ids)
    same_set = set(source_ids.tolist()) == set(ids.tolist())
    if same_set and len(source_ids) == len(ids):
        left, right = np.argsort(ids), np.argsort(source_ids)
        same_labels = np.array_equal(labels[left], source_labels[right])
        same_valid = np.array_equal(valid[left], source_valid[right])
    else:
        same_labels = same_valid = False
    result = dict(artifact_path=str(path.relative_to(ROOT)), artifact_sha256=sha256(path),
                  unique_source_ids=len(source_ids) == len(set(source_ids.tolist())),
                  sample_id_order_identical=bool(exact_order), sample_id_set_identical=bool(same_set),
                  ordered_labels_identical=bool(same_labels), ordered_valid_mask_identical=bool(same_valid),
                  sample_count=int(len(source_ids)))
    del source_labels, source_valid
    gc.collect()
    return result


def verify_aligned_per_image(rows: list[dict], probabilities: np.ndarray, labels: np.ndarray,
                             valid: np.ndarray, class_names: list[str]) -> dict:
    """Independently check every scalar in the corrected SpaceNet per-image table."""
    maximum = collections.defaultdict(float)
    mismatches = []
    nan_matches = 0
    for index, row in enumerate(rows):
        yy, vv = labels[index], valid[index]
        pr = np.moveaxis(probabilities[index], 0, -1)
        native = pr[vv]
        q, lab = native.astype(np.float64), yy[vv].astype(int)
        pred = q.argmax(1)
        cm = np.bincount(2 * lab + pred, minlength=4).reshape(2, 2)
        union = cm.sum(0) + cm.sum(1) - cm.diagonal()
        iou = np.divide(cm.diagonal(), union, out=np.full(2, np.nan), where=union > 0)
        decision = Bins(); decision.update(q.max(1), pred == lab)
        fg = np.clip(native[:, 1], 1e-12, 1 - 1e-7); by = lab == 1
        foreground = Bins(); foreground.update(fg, by)
        boundary = np.zeros_like(vv)
        change = (yy[1:] != yy[:-1]) & vv[1:] & vv[:-1]
        boundary[1:] |= change; boundary[:-1] |= change
        change = (yy[:, 1:] != yy[:, :-1]) & vv[:, 1:] & vv[:, :-1]
        boundary[:, 1:] |= change; boundary[:, :-1] |= change
        boundary = maximum_filter(boundary, size=3, mode='constant') & vv
        bp, bl = pr[boundary], yy[boundary]
        boundary_top = Bins(); boundary_top.update(bp.max(1), bp.argmax(1) == bl)
        boundary_fg = Bins(); boundary_fg.update(bp[:, 1], bl == 1)
        expected = dict(miou=float(np.nanmean(iou)), pixel_accuracy=float(cm.trace() / len(lab)),
                        nll=float(-np.log(np.maximum(q[np.arange(len(lab)), lab], 1e-12)).mean()),
                        brier=float(((q - np.eye(2)[lab]) ** 2).sum(1).mean()), ece_15=decision.ece(),
                        valid_pixels=int(vv.sum()), ignored_pixels=int((~vv).sum()),
                        foreground_ece_15=foreground.ece(),
                        foreground_nll=float((-(by * np.log(fg) + (~by) * np.log(1 - fg))).mean(dtype=np.float64)),
                        foreground_brier=float(((fg - by) ** 2).mean(dtype=np.float64)),
                        boundary_ece_15=boundary_top.ece(), boundary_foreground_ece_15=boundary_fg.ece())
        expected.update({f'iou_{name}': float(iou[c]) for c, name in enumerate(class_names)})
        for name, raw in row.items():
            if name == 'sample_id':
                continue
            actual = float(raw) if raw not in ('', 'None') else float('nan')
            value = expected[name]
            if np.isnan(value) and np.isnan(actual):
                nan_matches += 1
                continue
            difference = abs(value - actual)
            maximum[name] = max(maximum[name], difference)
            if not np.isfinite(difference) or difference > TOLERANCE:
                mismatches.append(dict(sample_id=row['sample_id'], metric=name, independent=value, reported=actual))
    return dict(rows_checked=len(rows), maximum_absolute_differences=dict(maximum),
                undefined_values_correctly_preserved=nan_matches, mismatches=mismatches,
                passed=not mismatches)


def verify(row: dict, inventory: list[dict], directory: Path) -> tuple[dict, list[dict]]:
    base = {name: row[name] for name in KEYS} | {'uq_method': 'dropout_off_same_mc_checkpoint'}
    path = directory / 'predictions.npz'
    original_metric_path = directory / 'metrics.json'
    alignment_path = directory / 'metric_alignment.json'
    alignment = json.loads(alignment_path.read_text()) if alignment_path.exists() else None
    metric_path = directory / alignment['canonical_metrics_file'] if alignment else original_metric_path
    completion_path = directory / 'completion.json'
    completion = json.loads(completion_path.read_text())
    saved_metrics = json.loads(metric_path.read_text())
    if 'metrics' in saved_metrics and isinstance(saved_metrics['metrics'], dict):
        saved_metrics = saved_metrics['metrics']
    contract = json.loads(row['data_contract'])
    names = contract['class_names']
    classes = len(names)
    ignore_index = contract.get('ignore_index')
    with np.load(path, allow_pickle=False) as archive:
        ids = archive['sample_id']
        labels = archive['label']
        probabilities = archive['probabilities']
        logits = archive['logits']
        valid = archive['valid_mask'].astype(bool)
        saved_prediction = archive['prediction']
    manifest_path = ROOT / f"reports/dataset_manifests/{row['dataset']}_actual_manifest.csv"
    with manifest_path.open() as f:
        test_ids = [r['sample_id'] for r in csv.DictReader(f) if r['split'] == 'test']
    confusion = np.zeros((classes, classes), dtype=np.int64)
    sums = collections.defaultdict(float)
    bins = {f'decision_{n}': Bins(n) for n in (10, 15, 30)}
    bins.update({f'class_probability_{i}': Bins() for i in range(classes)})
    is_space = row['dataset'] == 'spacenet7'
    if is_space:
        bins.update(foreground_probability_native_clip=Bins(), boundary_top_label=Bins(), boundary_building_probability=Bins())
    prediction_errors = 0
    max_softmax_error = 0.0
    max_probability_sum_error = 0.0
    images_without_boundary = 0
    finite = True
    range_ok = True
    valid_labels = True
    for start in range(0, len(labels), 8):
        yy = labels[start:start + 8]
        vv = valid[start:start + 8]
        raw = probabilities[start:start + 8]
        zz = logits[start:start + 8]
        finite &= bool(np.isfinite(raw).all() and np.isfinite(zz).all())
        range_ok &= bool(((raw >= 0) & (raw <= 1)).all())
        max_probability_sum_error = max(max_probability_sum_error, float(np.max(np.abs(raw.sum(axis=1, dtype=np.float64) - 1))))
        max_softmax_error = max(max_softmax_error, float(np.max(np.abs(softmax(zz.astype(np.float64), axis=1) - raw))))
        pr = np.moveaxis(raw, 1, -1)
        native_q = pr[vv]
        q = native_q.astype(np.float64)
        lab = yy[vv].astype(np.int64)
        valid_labels &= bool(((lab >= 0) & (lab < classes)).all())
        pred = q.argmax(axis=1)
        confidence = q.max(axis=1)
        prediction_errors += int(np.sum(pred != saved_prediction[start:start + 8][vv]))
        confusion += np.bincount(classes * lab + pred, minlength=classes ** 2).reshape(classes, classes)
        sums['n'] += len(lab)
        sums['ignored'] += int((~vv).sum())
        sums['nll'] += float(-np.log(np.maximum(q[np.arange(len(lab)), lab], 1e-12)).sum())
        sums['brier'] += float(((q - np.eye(classes)[lab]) ** 2).sum())
        sums['gap'] += float((confidence - (pred == lab)).sum())
        for n in (10, 15, 30):
            bins[f'decision_{n}'].update(confidence, pred == lab)
        for c in range(classes):
            bins[f'class_probability_{c}'].update(q[:, c], lab == c)
        if is_space:
            # Preserve native representability of upper clip (float32:
            # 0.9999998807907104), then use float64 only for reduction.
            foreground = np.clip(native_q[:, 1], 1e-12, 1 - 1e-7)
            building = lab == 1
            foreground_terms = -(building * np.log(foreground) + (~building) * np.log(1 - foreground))
            sums['foreground_nll'] += float(foreground_terms.sum(dtype=np.float64))
            sums['foreground_brier'] += float(((foreground - building) ** 2).sum(dtype=np.float64))
            bins['foreground_probability_native_clip'].update(foreground, building)
            boundary = np.zeros_like(vv)
            horizontal = (yy[:, :, 1:] != yy[:, :, :-1]) & vv[:, :, 1:] & vv[:, :, :-1]
            vertical = (yy[:, 1:, :] != yy[:, :-1, :]) & vv[:, 1:, :] & vv[:, :-1, :]
            boundary[:, :, 1:] |= horizontal
            boundary[:, :, :-1] |= horizontal
            boundary[:, 1:, :] |= vertical
            boundary[:, :-1, :] |= vertical
            boundary = maximum_filter(boundary, size=(1, 3, 3), mode='constant') & vv
            images_without_boundary += int((~boundary.reshape(len(yy), -1).any(axis=1)).sum())
            sums['boundary'] += int(boundary.sum())
            bp, by = pr[boundary], yy[boundary]
            bins['boundary_top_label'].update(bp.max(axis=1), bp.argmax(axis=1) == by)
            bins['boundary_building_probability'].update(bp[:, 1], by == 1)
    intersection = confusion.diagonal()
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    iou = np.divide(intersection, union, out=np.full(classes, np.nan), where=union > 0)
    metrics = dict(miou=float(np.nanmean(iou)), pixel_accuracy=float(intersection.sum() / sums['n']),
                   nll=sums['nll'] / sums['n'], brier=sums['brier'] / sums['n'],
                   mean_confidence_minus_accuracy=sums['gap'] / sums['n'],
                   per_class_iou={name: float(iou[i]) for i, name in enumerate(names)},
                   valid_pixels=int(sums['n']), ignored_pixels=int(sums['ignored']),
                   confusion_matrix=confusion.tolist(),
                   classwise_calibration={name: {'ece_15': bins[f'class_probability_{i}'].ece()} for i, name in enumerate(names)})
    for n in (10, 15, 30):
        metrics[f'ece_{n}'] = bins[f'decision_{n}'].ece()
    if is_space:
        metrics.update(foreground_ece_15=bins['foreground_probability_native_clip'].ece(),
                       foreground_nll=sums['foreground_nll'] / sums['n'],
                       foreground_brier=sums['foreground_brier'] / sums['n'],
                       boundary_calibration=dict(radius_pixels=1, pixel_count=int(sums['boundary']),
                                                 ece_15=bins['boundary_top_label'].ece(),
                                                 foreground_ece_15=bins['boundary_building_probability'].ece()),
                       images_without_boundary=images_without_boundary)
    computed_leaves, saved_leaves = scalar_leaves(metrics), scalar_leaves(saved_metrics)
    differences = {k: computed_leaves[k] - v for k, v in saved_leaves.items() if k in computed_leaves}
    missing_recomputed = sorted(set(saved_leaves) - set(computed_leaves))
    mismatches = {k: v for k, v in differences.items() if abs(v) > TOLERANCE or not np.isfinite(v)}
    order = np.argsort(ids)
    expected_mask = np.ones_like(valid) if ignore_index is None else labels != ignore_index
    checks = dict(full_test_id_set=set(ids.tolist()) == set(test_ids),
                  expected_test_sample_count=len(ids) == len(test_ids),
                  unique_ids=len(ids) == len(set(ids.tolist())),
                  ids_count=int(len(ids)), test_manifest_sha256=sha256(manifest_path),
                  mask_matches_ignore_contract=bool(np.array_equal(valid, expected_mask)),
                  valid_labels_in_range=bool(valid_labels), finite_probabilities_and_logits=bool(finite),
                  probability_range=bool(range_ok),
                  max_logit_probability_error=max_softmax_error,
                  logit_probability_agreement=max_softmax_error <= TOLERANCE,
                  probability_sum_max_error=max_probability_sum_error,
                  probability_sum_agreement=max_probability_sum_error <= TOLERANCE,
                  argmax_vs_saved_prediction_pixel_difference=prediction_errors,
                  saved_prediction_agreement=prediction_errors == 0,
                  confusion_matrix_exact=np.array_equal(confusion, saved_metrics['confusion_matrix']),
                  ordered_labels_sha256=array_sha256(labels[order]),
                  ordered_valid_mask_sha256=array_sha256(valid[order]),
                  all_reported_scalar_metrics_recomputed=not missing_recomputed,
                  no_metric_mismatches=not mismatches)
    pair_checks = {}
    for method in ('deterministic', 'mc_dropout'):
        matches = [r for r in inventory if key(r) == key(row) and r['uq_method'] == method]
        if len(matches) != 1:
            raise ValueError(f'{key(row)} requires one {method} comparator: got {len(matches)}')
        pair_checks[method] = source_pair_checks(matches[0], ids, labels, valid)
    checkpoint_path = Path(row['checkpoint_path'])
    checkpoint_hash = sha256(checkpoint_path)
    supplied_checkpoint_hashes = find_values(completion, 'checkpoint_sha256')
    checks.update(checkpoint_sha256=checkpoint_hash,
                  checkpoint_matches_mc_inventory=checkpoint_hash == row['checkpoint_sha256'],
                  completion_records_same_checkpoint=bool(supplied_checkpoint_hashes) and all(h == checkpoint_hash for h in supplied_checkpoint_hashes))
    artifact_hash = sha256(path)
    metrics_hash = sha256(metric_path)
    original_metrics_hash = sha256(original_metric_path) if alignment else metrics_hash
    evaluation_sources = completion.get('code_sha256', {})
    source_checks = {name: (ROOT / name).is_file() and sha256(ROOT / name) == expected
                     for name, expected in evaluation_sources.items()}
    checks.update(completion_prediction_hash_matches=completion.get('prediction_sha256') == artifact_hash,
                  completion_metrics_hash_matches=completion.get('metrics_sha256') == original_metrics_hash,
                  evaluation_source_hashes_match=bool(source_checks) and all(source_checks.values()),
                  completion_training_performed_false=completion.get('training_performed') is False,
                  completion_dropout_off_true=completion.get('dropout_off') is True,
                  completion_runtime_checks_pass=bool(completion.get('checks')) and all(v is True for v in completion['checks'].values()))
    alignment_checks = {}
    aligned_per_image = None
    if alignment:
        known_hashes = {'predictions.npz': artifact_hash, 'metrics.json': original_metrics_hash,
                        metric_path.name: metrics_hash, 'completion.json': sha256(completion_path)}
        alignment_checks = {
            name: known_hashes.get(name, '') == expected if name in known_hashes else sha256(directory / name) == expected
            for name, expected in alignment['artifact_hashes'].items()
        }
        alignment_checks.update({
            'alignment_records_all_original_and_canonical_files':
                set(('predictions.npz', 'metrics.json', 'completion.json', 'per_image_metrics.csv',
                     alignment['canonical_metrics_file'], alignment['canonical_per_image_file'])) <= set(alignment['artifact_hashes']),
            'original_confusion_matrix_preserved': alignment['original_confusion_matrix'] == json.loads(original_metric_path.read_text())['confusion_matrix'],
            'canonical_confusion_matrix_exact': np.array_equal(confusion, alignment['canonical_confusion_matrix']),
            'original_files_unchanged_recorded': alignment.get('original_files_unchanged') is True,
            'no_inference_or_training_recorded': alignment.get('no_inference_or_training') is True,
            'cpu_softmax_matches_export_recorded': alignment.get('cpu_recomputed_softmax_max_export_error') == 0,
            'cpu_argmax_matches_export_recorded': alignment.get('cpu_recomputed_argmax_export_difference') == 0,
            'alignment_source_hashes_match': bool(alignment.get('code_sha256')) and all(
                sha256(ROOT / name) == expected for name, expected in alignment['code_sha256'].items()),
        })
        with (directory / alignment['canonical_per_image_file']).open() as f:
            per_image_rows = list(csv.DictReader(f))
        alignment_checks['canonical_per_image_ids_exact'] = [r['sample_id'] for r in per_image_rows] == ids.tolist()
        aligned_per_image = verify_aligned_per_image(per_image_rows, probabilities, labels, valid, names)
        alignment_checks['canonical_per_image_metrics_independently_verified'] = aligned_per_image['passed']
        checks['metric_alignment_hashes_and_metadata_valid'] = all(v is True for v in alignment_checks.values())
    pair_ok = all(v is not False for evidence in pair_checks.values() for v in evidence.values())
    passed = all(v is not False for v in checks.values()) and pair_ok
    result = dict(base, status='PASS' if passed else 'FAIL', artifact_path=str(path.relative_to(ROOT)),
                  artifact_sha256=artifact_hash, metrics_path=str(metric_path.relative_to(ROOT)),
                  metrics_sha256=metrics_hash, completion_path=str(completion_path.relative_to(ROOT)),
                  completion_sha256=sha256(completion_path), verifier_sha256=sha256(Path(__file__)),
                  tolerance=TOLERANCE, metrics=metrics, metric_differences=differences,
                  metric_mismatches_over_tolerance=mismatches, missing_recomputed_metrics=missing_recomputed,
                  checks=checks, paired_source_checks=pair_checks, evaluation_source_checks=source_checks,
                  original_metrics_path=str(original_metric_path.relative_to(ROOT)), original_metrics_sha256=original_metrics_hash,
                  metric_alignment_path=str(alignment_path.relative_to(ROOT)) if alignment else None,
                  metric_alignment_sha256=sha256(alignment_path) if alignment else None,
                  metric_alignment_checks=alignment_checks, aligned_per_image_verification=aligned_per_image)
    bin_rows = [record for kind, b in bins.items() for record in b.records(base, kind)]
    del labels, probabilities, logits, valid, saved_prediction
    gc.collect()
    return result, bin_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--force', action='store_true', help='Ignore successful cache entries.')
    args = parser.parse_args()
    inventory = list(csv.DictReader((OLD / 'actual_artifacts.csv').open()))
    expected = [r for r in inventory if r['task'] == 'segmentation' and r['uq_method'] == 'mc_dropout']
    results, all_bins, pending = [], [], []
    for row in expected:
        directory = ARTIFACT_ROOT / row['dataset'] / row['model'] / row['adaptation'] / ('seed' + row['seed'])
        if not all((directory / name).is_file() for name in ('completion.json', 'metrics.json', 'predictions.npz')):
            pending.append('/'.join(key(row)))
            continue
        cache_path = OUT / 'verification_cache' / ('_'.join(key(row)) + '.json')
        fingerprint_names = ['completion.json', 'metrics.json', 'predictions.npz']
        if (directory / 'metric_alignment.json').exists():
            alignment = json.loads((directory / 'metric_alignment.json').read_text())
            fingerprint_names += ['metric_alignment.json', alignment['canonical_metrics_file'], alignment['canonical_per_image_file']]
        fingerprints = {name: dict(size=(directory / name).stat().st_size, mtime_ns=(directory / name).stat().st_mtime_ns)
                        for name in fingerprint_names}
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        if (not args.force and cached.get('version') == VERSION and cached.get('fingerprints') == fingerprints
                and cached.get('result', {}).get('verifier_sha256') == sha256(Path(__file__))
                and cached.get('result', {}).get('status') == 'PASS'):
            result, bin_rows = cached['result'], cached['bins']
            print('CACHED', '/'.join(key(row)), result['status'], flush=True)
        else:
            print('VERIFY', '/'.join(key(row)), flush=True)
            result, bin_rows = verify(row, inventory, directory)
            write_json(cache_path, dict(version=VERSION, fingerprints=fingerprints, result=result, bins=bin_rows))
            print('RESULT', '/'.join(key(row)), result['status'], result['metric_mismatches_over_tolerance'], flush=True)
        results.append(result)
        all_bins.extend(bin_rows)
        write_json(OUT / 'segmentation_off_verification.json', results)
        write_csv(OUT / 'segmentation_off_reliability_bins.csv', all_bins)
        flat_rows = [dict({k: r[k] for k in (*KEYS, 'uq_method', 'status', 'artifact_path')}, **scalar_leaves(r['metrics'])) for r in results]
        write_csv(OUT / 'segmentation_off_recomputed_metrics.csv', flat_rows)
    summary = dict(expected_count=len(expected), verified_count=len(results),
                   passed=sum(r['status'] == 'PASS' for r in results), failed=sum(r['status'] == 'FAIL' for r in results),
                   pending=pending, complete=not pending and len(results) == 12,
                   maximum_absolute_metric_difference=max((abs(d) for r in results for d in r['metric_differences'].values()), default=None),
                   verifier_path=str(Path(__file__).relative_to(ROOT)), verifier_sha256=sha256(Path(__file__)),
                   no_project_metric_imports=True,
                   foreground_convention='Clip and logarithm retain native probability dtype; float64 accumulation; no log-softmax retransform for dropout-off logits export.',
                   paired_verification='Direct complete ID/label/mask arrays and SHA256 from old deterministic and MC source NPZs.')
    write_json(OUT / 'segmentation_off_verification_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if summary['failed'] or (args.require_complete and pending) else 0


if __name__ == '__main__':
    raise SystemExit(main())
