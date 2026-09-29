"""Summarize stored costs and design counts; do not execute proposed studies."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import statistics
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
master_path = ROOT / 'reports/core_rq_completion_20260921/published/tables/thesis_master_results.csv'
archive_path = ROOT / 'research_data/manifest.parquet'
with master_path.open() as f:
    master = list(csv.DictReader(f))
individual = [r for r in master if r['record_type'] == 'INDIVIDUAL_SEED']
trained = [r for r in individual if r['uq_method'] in ['deterministic', 'mc_dropout']]
assert len(trained) == len({r['run_id'] for r in trained}) == 72
archive = pq.read_table(archive_path).to_pylist()
groups = []
for method in ['deterministic', 'mc_dropout']:
    for dataset in ['eurosat', 'treesatai', 'cloudsen12', 'spacenet7']:
        for adaptation in ['frozen', 'full_finetune']:
            rs = [r for r in trained if r['uq_method'] == method and r['dataset'] == dataset and r['adaptation'] == adaptation]
            values = [float(r['training_seconds']) for r in rs]
            groups.append({'method': method, 'dataset': dataset, 'adaptation': adaptation, 'runs': len(rs),
                           'cumulative_run_wall_hours': sum(values) / 3600,
                           'median_run_minutes': statistics.median(values) / 60,
                           'min_run_minutes': min(values) / 60, 'max_run_minutes': max(values) / 60})
role_size = {}
for r in archive:
    d = role_size.setdefault(r['role'], {'files': 0, 'bytes': 0})
    d['files'] += 1
    d['bytes'] += r['size_bytes']
for d in role_size.values():
    d['GiB'] = d['bytes'] / 2**30
classification_bytes = sum(r['size_bytes'] for r in archive if r['task'] == 'classification')
segmentation_bytes = sum(r['size_bytes'] for r in archive if r['task'] == 'segmentation')
packages = [
    {'package': 'A_existing_predictions', 'new_training_min': 0, 'new_training_max': 0,
     'new_backbone_dataset_evaluations': 0, 'classification_prediction_objects': 68,
     'segmentation_primary_prediction_objects': 32, 'note': 'Full-image statistics and common-subset pixel diagnostics; no new model inference.'},
    {'package': 'B_pooled_CKA', 'new_training_min': 0, 'new_training_max': 0,
     'existing_paired_comparisons': 21, 'optional_missing_embedding_evaluations': 6,
     'optional_unique_encoder_evaluations_if_frozen_equivalence_verified': 4},
    {'package': 'C_corruption_D_only', 'new_training_min': 0, 'new_training_max': 0,
     'new_backbone_dataset_evaluations': 8, 'new_head_dataset_evaluations': 8,
     'note': 'EuroSAT 4 cells times 2 new severities; each evaluation covers 2714 images.'},
    {'package': 'C_corruption_all_methods', 'new_training_min': 0, 'new_training_max': 0,
     'new_backbone_dataset_evaluations': 32, 'new_head_dataset_evaluations': 272,
     'unique_backbone_evaluations_if_frozen_cross_checkpoint_equivalence_verified': 20,
     'breakdown': {'deterministic_member_backbone': 24, 'mc_backbone': 8,
                   'deterministic_member_head': 24, 'mc_stochastic_head': 240, 'mc_off_head': 8},
     'note': 'Conservative per-checkpoint encoding: 4 cells x 2 severities x (3 D members + 1 MC model); D42 reused from DE member; TS offline. Frozen D/MC encoders may share features only after state/preprocessing/extractor equivalence checks, reducing 32 to 20. The 272 head passes are not 272 backbone passes.'},
    {'package': 'D_module_factorial_2FM', 'new_training_min': 24, 'new_training_max': 24,
     'total_training_positions': 24, 'note': 'Current old endpoints use different recipes; budget all 24 positions under the new common protocol. Reuse requires actual matching evidence, not hypothetical endpoints.'},
    {'package': 'D_module_factorial_1FM', 'new_training_min': 12, 'new_training_max': 12,
     'total_training_positions': 12},
    {'package': 'E_toy', 'new_training_min': 0, 'new_training_max': 0,
     'primary_factorial_conditions': 4, 'weighted_analytic_beta_terms_two_strata': 888,
     'optional_sampling_condition_replicates': 80, 'optional_sampled_beta_posteriors': 160,
     'note': 'Primary analysis enumerates binomial counts analytically: 2 ambiguity x 2 strata x (21+201) count states. Optional 20 paired-data illustrations are not FM training runs.'},
    {'package': 'F_support_pilot', 'new_training_min': 6, 'new_training_max': 6,
     'total_unique_training_positions': 6, 'model_input_states': 18,
     'conditional_legacy_recipe_panopticon_eurosat_frozen_mc_new_runs': 5,
     'conditional_legacy_recipe_dofa_eurosat_frozen_mc_new_runs': 3,
     'note': 'Budget all 6 under common optimizer-step control. Legacy reuse changes/limits the estimand and requires exact run matching. Panopticon EuroSAT frozen MC has only seed42; DOFA has 42/43/44. Subset and optimizer variation remain combined.'},
    {'package': 'F_support_crossed', 'new_training_min': 12, 'new_training_max': 12,
     'total_unique_training_positions': 12, 'model_input_states': 36,
     'conditional_legacy_recipe_panopticon_eurosat_frozen_mc_new_runs': 11,
     'conditional_legacy_recipe_dofa_eurosat_frozen_mc_new_runs': 9,
     'note': 'Budget all 12 under common optimizer-step control. 3 full-support optimizer seeds plus 3 subsets x 3 optimizer seeds low support; not 18. Conditional old-recipe reuse requires exact matching and carries its claim limitations. New training feature extraction may be needed.'},
]
for p in packages:
    p['new_training_fraction_of_72'] = [p['new_training_min'] / 72, p['new_training_max'] / 72]
result = {
    'baseline': {'unique_neural_training_runs': 72,
                 'deterministic_runs': 48, 'mc_runs': 24,
                 'deterministic_run_wall_hours': sum(float(r['training_seconds']) for r in trained if r['uq_method'] == 'deterministic') / 3600,
                 'mc_run_wall_hours': sum(float(r['training_seconds']) for r in trained if r['uq_method'] == 'mc_dropout') / 3600,
                 'total_cumulative_run_wall_hours': sum(float(r['training_seconds']) for r in trained) / 3600,
                 'warning': 'Stored cumulative run wall times; not GPU-hours or project elapsed time. DE reuses D runs; do not double count.'},
    'historical_training_strata': groups,
    'archive': {'records': len(archive), 'classification_GiB': classification_bytes / 2**30,
                'segmentation_GiB': segmentation_bytes / 2**30, 'role_sizes': role_size,
                'note': 'Archived compressed bytes, not peak RAM, exact A-package bytes, or new storage requirement. Off controls are stored outside this archive.'},
    'packages': packages,
    'inference_cost_scope': 'C counts exact dataset-wide operations under head-only dropout feature reuse; no measured new end-to-end seconds.',
    'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [master_path, archive_path]},
}
(OUT / 'workload_estimates.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
with (OUT / 'historical_training_costs.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(groups[0])); w.writeheader(); w.writerows(groups)
print(json.dumps(result['baseline'], ensure_ascii=False, indent=2))
print('PACKAGES', len(packages), 'CLASSIFICATION_ARCHIVE_GiB', result['archive']['classification_GiB'], 'SEGMENTATION_ARCHIVE_GiB', result['archive']['segmentation_GiB'])
