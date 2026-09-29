"""Read existing evidence tables; do not run models or change sealed results."""
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/core_rq_completion_20260921'
inputs = {}


def read(path):
    path = Path(path)
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:
        return list(csv.DictReader(f))


master = read(BASE / 'published/tables/thesis_master_results.csv')
three = read(BASE / 'mc_dropout_three_way.csv')
mc = read(ROOT / 'reports/mc_dropout_results.csv')
checks = read(BASE / 'checks.csv')
coverage = read(BASE / 'coverage_matrix.csv')
individual = [r for r in master if r['record_type'] == 'INDIVIDUAL_SEED']
det = [r for r in individual if r['uq_method'] == 'deterministic']
key = lambda r: (r['dataset'], r['model'], r['adaptation'], r['seed'])
det_index = {key(r): r for r in det}
ts = [r for r in individual if r['uq_method'] == 'temperature_scaling']

result = {
    'scope': 'Independent table arithmetic and source hashes; not a new full-prediction audit or new experiment.',
    'audit_status_counts': dict(Counter(r['status'] for r in checks)),
    'coverage_status_counts': dict(Counter(r['coverage_status'] for r in coverage)),
    'deterministic_runs': len(det),
    'mc_matched_controls': len(three),
    'temperature_scaling': {
        'n': len(ts),
        'ece_improved': sum(float(r['ece_15']) < float(det_index[key(r)]['ece_15']) for r in ts),
        'accuracy_unchanged': sum(float(r['accuracy']) == float(det_index[key(r)]['accuracy']) for r in ts),
    },
}
for task in ['classification', 'segmentation']:
    rows = [r for r in three if r['task'] == task]
    result[task + '_mc_minus_off'] = {
        'n': len(rows),
        **{metric + '_improved': sum(float(r['MC_minus_Off_' + metric]) < 0 for r in rows)
           for metric in ['ece_15', 'nll', 'brier']},
    }
result['cloud_dofa_frozen_seed42_ece'] = {
    k: float(r[k]) for r in three
    if key(r) == ('cloudsen12', 'dofa', 'frozen', '42')
    for k in ['D_ece_15', 'Off_ece_15', 'MC_ece_15',
              'MC_minus_D_ece_15', 'MC_minus_Off_ece_15', 'Off_minus_D_ece_15']
}
adaptation = []
for dataset in ['eurosat', 'treesatai', 'cloudsen12', 'spacenet7']:
    for model in ['dofa', 'panopticon']:
        metric = 'accuracy' if dataset == 'eurosat' else 'macro_f1' if dataset == 'treesatai' else 'miou'
        record = {'dataset': dataset, 'model': model, 'n': 3, 'primary_metric': metric}
        for m in [metric, 'ece_15', 'nll', 'brier']:
            deltas = [float(det_index[(dataset, model, 'full_finetune', s)][m]) -
                      float(det_index[(dataset, model, 'frozen', s)][m]) for s in ['42', '43', '44']]
            record[m] = {'mean': statistics.mean(deltas), 'sample_sd': statistics.stdev(deltas),
                         'min': min(deltas), 'max': max(deltas), 'values': deltas}
        adaptation.append(record)
result['adaptation_full_minus_frozen'] = adaptation
ensemble = []
for r in master:
    if r['record_type'] != 'ENSEMBLE' or r['dataset'] != 'spacenet7':
        continue
    item = {k: r[k] for k in ['dataset', 'model', 'adaptation']}
    for metric in ['iou_building', 'miou', 'ece_15', 'nll', 'brier']:
        dmean = statistics.mean(float(det_index[(r['dataset'], r['model'], r['adaptation'], s)][metric])
                                for s in ['42', '43', '44'])
        item[metric] = {'ensemble': float(r[metric]), 'member_metric_mean': dmean,
                        'delta': float(r[metric]) - dmean}
    ensemble.append(item)
result['spacenet7_ensemble_minus_member_mean'] = ensemble
result['deterministic_per_seed_ranges'] = {}
for dataset, metric in [('spacenet7', 'iou_building'), ('spacenet7', 'pixel_accuracy')]:
    values = [float(r[metric]) for r in det if r['dataset'] == dataset]
    result['deterministic_per_seed_ranges'][dataset + '_' + metric] = [min(values), max(values)]
tree_means = [float(r['macro_f1']) for r in master if r['dataset'] == 'treesatai'
              and r['uq_method'] == 'deterministic' and r['record_type'] == 'MEAN_STD']
result['treesatai_deterministic_cell_mean_macro_f1_range'] = [min(tree_means), max(tree_means)]
result['eurosat_mc_seed42_descriptive_proxy_example'] = [
    {**{k: r[k] for k in ['model', 'adaptation', 'seed']},
     **{k: float(r[k]) for k in ['mean_mi_style_disagreement', 'mean_expected_predictive_entropy',
                                'mc_accuracy', 'mc_nll', 'mc_brier']}}
    for r in mc if r['dataset'] == 'eurosat' and r['seed'] == '42'
]
result['input_sha256'] = inputs
result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT / 'independent_evidence_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in
                  ['input_sha256', 'adaptation_full_minus_frozen', 'spacenet7_ensemble_minus_member_mean']},
                 ensure_ascii=False, indent=2))
