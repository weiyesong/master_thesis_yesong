"""Read metadata and representative headers; no training/inference/new metrics."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re
import zipfile
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'research_data'
rows = pq.read_table(ARCHIVE / 'manifest.parquet').to_pylist()
inventory = []
for r in rows:
    p = ARCHIVE / r['archive_path']
    inventory.append({k: r[k] for k in ['artifact_id', 'role', 'dataset', 'model', 'adaptation',
                                      'uq_method', 'seed', 'sample_count', 'archive_path']} |
                     {'exists': p.is_file(), 'size_matches': p.is_file() and p.stat().st_size == r['size_bytes']})

roles = ['classification_deterministic_predictions', 'classification_mc_raw_passes',
         'classification_ensemble_members', 'classification_deterministic_embeddings',
         'segmentation_representations', 'segmentation_mc_full_maps',
         'segmentation_ensemble_member_subset']
header_checks = []
for role in roles:
    r = next(r for r in rows if r['role'] == role)
    p = ARCHIVE / r['archive_path']
    if p.suffix == '.parquet':
        schema = pq.read_schema(p)
        actual = {field.name: str(field.type) for field in schema}
        expected = {x['name']: x['type'] for x in json.loads(r['array_schema_json'])['columns']}
        matches = actual == expected
    else:
        actual = {}
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if not name.endswith('.npy'):
                    continue
                with z.open(name) as f:
                    version = np.lib.format.read_magic(f)
                    if version == (1, 0):
                        shape, order, dtype = np.lib.format.read_array_header_1_0(f)
                    else:
                        shape, order, dtype = np.lib.format.read_array_header_2_0(f)
                actual[name[:-4]] = {'shape': list(shape), 'fortran_order': order, 'dtype': str(dtype)}
        expected = json.loads(r['array_schema_json'])['arrays']
        matches = actual == expected
    header_checks.append({'role': role, 'path': str(p.relative_to(ROOT)),
                          'header_matches_manifest': matches, 'actual_schema': actual})

sources = [
    'reports/core_rq_completion_20260921/EO_UQ_Thesis_Conversation_Dossier_for_Codex.md',
    'EO_FM_UQ_Core_RQ_Audit_Standard.md', 'research_data/manifest.parquet',
    'research_data/README.md', 'reports/pre_uq_protocol_freeze.md',
    'reports/core_rq_completion_20260921/checks.csv',
    'reports/core_rq_completion_20260921/coverage_matrix.csv',
    'reports/core_rq_completion_20260921/published/tables/thesis_master_results.csv',
    'reports/core_rq_completion_20260921/mc_dropout_three_way.csv',
    'reports/core_rq_completion_20260921/provenance_followup.md',
    'reports/thesis_readiness_20260926/independent_evidence_check.json',
]
sha = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources}
def csvrows(p):
    with (ROOT / p).open() as f:
        return list(csv.DictReader(f))
master = csvrows('reports/core_rq_completion_20260921/published/tables/thesis_master_results.csv')
checks = csvrows('reports/core_rq_completion_20260921/checks.csv')
coverage = csvrows('reports/core_rq_completion_20260921/coverage_matrix.csv')
three = csvrows('reports/core_rq_completion_20260921/mc_dropout_three_way.csv')

# Inspect filenames only in the admitted results/report/script trees. Absence is
# bounded by this search, not a claim that no such work exists elsewhere.
pattern = re.compile(r'aurc|auroc|risk[_-]?coverage|error[_-]?detect|(?:^|[_-])cka(?:[_\.-]|$)|coupling|corruption[_-]results|label[_-]scarcity', re.I)
matches = []
for folder in ['reports', 'scripts', 'results']:
    for p in (ROOT / folder).rglob('*'):
        if p.is_file() and pattern.search(p.name) and 'thesis_dossier_assessment_20260926' not in str(p):
            matches.append(str(p.relative_to(ROOT)))
result = {
    'scope': 'Manifest existence/size and 7 representative real-file header checks only; not a 65 GiB hash or probability audit.',
    'archive_records': len(rows), 'role_counts': dict(Counter(r['role'] for r in rows)),
    'all_paths_and_sizes_match': all(r['exists'] and r['size_matches'] for r in inventory),
    'header_checks': header_checks, 'archive_inventory': inventory,
    'core_audit_status_counts': dict(Counter(r['status'] for r in checks)),
    'core_coverage_status_counts': dict(Counter(r['coverage_status'] for r in coverage)),
    'individual_results_by_method': dict(Counter(r['uq_method'] for r in master if r['record_type'] == 'INDIVIDUAL_SEED')),
    'ensemble_groups': sum(r['record_type'] == 'ENSEMBLE' for r in master),
    'matched_off_controls_by_task': dict(Counter(r['task'] for r in three)),
    'historical_and_current_archive_ts_counts': dict(Counter(r['dataset'] for r in rows if r['role'] == 'classification_temperature_scaling_predictions')),
    'extension_result_filename_search': {'roots': ['reports', 'scripts', 'results'], 'pattern': pattern.pattern, 'matches': matches},
    'source_sha256': sha,
}
(OUT / 'evidence_inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ['header_checks', 'archive_inventory', 'source_sha256']}, ensure_ascii=False, indent=2))
print('HEADER_CHECKS', [(r['role'], r['header_matches_manifest']) for r in header_checks])
assert result['all_paths_and_sizes_match']
assert all(r['header_matches_manifest'] for r in header_checks)
