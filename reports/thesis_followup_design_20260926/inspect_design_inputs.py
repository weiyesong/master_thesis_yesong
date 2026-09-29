"""Read metadata and headers only; no new model or research metric execution."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import zipfile
import numpy.lib.format as npformat
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
archive_path = ROOT / 'research_data/manifest.parquet'
archive = pq.read_table(archive_path).to_pylist()
master_path = ROOT / 'reports/core_rq_completion_20260921/published/tables/thesis_master_results.csv'
with master_path.open() as f:
    master = list(csv.DictReader(f))
threeway_path = ROOT / 'reports/core_rq_completion_20260921/mc_dropout_three_way.csv'
with threeway_path.open() as f:
    threeway = list(csv.DictReader(f))

class_roles = {'classification_deterministic_predictions', 'classification_mc_saved_summaries',
               'classification_temperature_scaling_predictions', 'classification_ensemble_mean'}
seg_roles = {'segmentation_deterministic_predictions', 'segmentation_mc_full_maps', 'segmentation_ensemble_mean'}
objects = []
for row in archive:
    classification = row['role'] in class_roles
    segmentation = row['role'] in seg_roles and (row['seed'] == 42 or row['seed'] is None)
    if not (classification or segmentation):
        continue
    if row['uq_method'] == 'temperature_scaling' and row['dataset'] != 'eurosat':
        continue
    path = ROOT / 'research_data' / row['archive_path']
    assert path.is_file()
    assert path.stat().st_size == row['size_bytes']
    objects.append({'task': row['task'], 'dataset': row['dataset'], 'model': row['model'],
                    'adaptation': row['adaptation'], 'method': row['uq_method'],
                    'seed': row['seed'] if row['seed'] is not None else '',
                    'member_seeds': json.dumps(row['member_seeds']),
                    'representative_prediction_path': str(path.relative_to(ROOT)),
                    'path_exists': True, 'sample_count_from_manifest': row['sample_count'],
                    'source': str(archive_path.relative_to(ROOT))})
for row in threeway:
    if row['task'] == 'segmentation' and row['seed'] != '42':
        continue
    path = ROOT / row['output_path']
    assert path.is_file()
    objects.append({'task': row['task'], 'dataset': row['dataset'], 'model': row['model'],
                    'adaptation': row['adaptation'], 'method': 'mc_dropout_off',
                    'seed': row['seed'], 'member_seeds': '[]',
                    'representative_prediction_path': str(path.relative_to(ROOT)),
                    'path_exists': True, 'sample_count_from_manifest': int(row['sample_count']),
                    'source': str(threeway_path.relative_to(ROOT))})
counts = Counter(row['task'] for row in objects)
assert counts == {'classification': 68, 'segmentation': 32}, counts
keys = [(r['dataset'], r['model'], r['adaptation'], r['method'], str(r['seed'])) for r in objects]
assert len(set(keys)) == len(keys)
with (OUT / 'planned_A_prediction_objects.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(objects[0]))
    writer.writeheader()
    writer.writerows(objects)

roles = ['classification_mc_raw_passes', 'classification_ensemble_members',
         'segmentation_mc_full_maps', 'segmentation_mc_raw_subset',
         'segmentation_ensemble_member_subset', 'segmentation_representations']
headers = []
for role in roles:
    row = next(r for r in archive if r['role'] == role)
    path = ROOT / 'research_data' / row['archive_path']
    actual = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('.npy'):
                continue
            with z.open(name) as f:
                version = npformat.read_magic(f)
                shape, fortran, dtype = npformat._read_array_header(f, version)
            actual[name[:-4]] = {'shape': list(shape), 'dtype': str(dtype), 'fortran_order': fortran}
    advertised = json.loads(row['array_schema_json'])['arrays']
    assert actual == advertised, (role, actual, advertised)
    headers.append({'role': role, 'path': str(path.relative_to(ROOT)), 'actual_arrays': actual,
                    'matches_manifest_schema': True, 'actual_size_bytes': path.stat().st_size})

mc_euro = [r for r in master if r['record_type'] == 'INDIVIDUAL_SEED'
           and r['uq_method'] == 'mc_dropout' and r['dataset'] == 'eurosat']
seed_sets = {f'{m}/{a}': sorted(int(r['seed']) for r in mc_euro if r['model'] == m and r['adaptation'] == a)
             for m in ['dofa', 'panopticon'] for a in ['frozen', 'full_finetune']}
assert seed_sets['panopticon/frozen'] == [42]
assert seed_sets['dofa/frozen'] == [42, 43, 44]
result = {'scope': 'Metadata, path/byte checks and representative actual NPZ headers only; no new metrics, inference, training, or full archive hash audit.',
          'planned_objects': dict(counts), 'all_100_representative_paths_exist': True,
          'euro_mc_seed_sets_direct_from_master': seed_sets, 'representative_headers': headers,
          'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [archive_path, master_path, threeway_path]}}
(OUT / 'design_input_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'objects': dict(counts), 'actual_headers_checked': len(headers), 'mc_euro_seed_sets': seed_sets}, ensure_ascii=False))
