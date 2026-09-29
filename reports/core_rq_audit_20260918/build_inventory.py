"""Read-only inventory for the 29-item core-RQ audit. Writes only this folder."""
from pathlib import Path
import csv, json, hashlib, itertools, collections
import yaml

ROOT = Path('/workspace')
OUT = Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text())
def rel(p): return str(Path(p).resolve().relative_to(ROOT))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def model_config(cfg):
    active=cfg.get('active_experiment')
    if isinstance(active,dict): return active.get('model',{})
    return next((e.get('model',{}) for e in cfg.get('experiments',[]) if e.get('name')==active), cfg.get('experiments',[{}])[0].get('model',{}))
def csvout(name, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
master=list(csv.DictReader((ROOT/'reports/thesis_master_results.csv').open()))
snapshot=read(OUT/'snapshot.json')['snapshot_id']
actual=[]; runtime=[]
for r in master:
    if r['record_type'] not in ('INDIVIDUAL_SEED','ENSEMBLE'): continue
    method=r['uq_method']; cp=r['checkpoint_path']; meta={}
    if method=='deterministic':
        d=Path(cp).parent/'predictions/test/deterministic'
        artifact=d/('predictions.parquet' if r['task']=='classification' else 'predictions.npz')
        manifest=d/'manifest.json'
    else:
        manifest=ROOT/r['source_paths'].split(';')[-1]; d=manifest.parent
        names={'temperature_scaling':'test_predictions.parquet','mc_dropout':'predictions.parquet' if r['task']=='classification' else 'aggregate_uncertainty_maps.npz','deep_ensemble':'ensemble_predictions.parquet' if r['task']=='classification' else 'ensemble_predictions.npz'}
        artifact=d/names[method]
    if manifest.exists():meta=read(manifest)
    row={k:r[k] for k in ('task','dataset','model','adaptation','uq_method','seed','contributing_seeds','record_type','run_id','checkpoint_path','checkpoint_sha256','checkpoint_selection_criterion','metric_semantics','source_paths')}
    row.update(artifact_path=rel(artifact),artifact_exists=artifact.is_file(),artifact_bytes=artifact.stat().st_size if artifact.is_file() else 0,manifest_path=rel(manifest),manifest_sha256=sha(manifest) if manifest.exists() else '',audit_snapshot=snapshot)
    if cp and ';' not in cp:
        rd=Path(cp).parent; config=rd/'resolved_config.yaml'; cs=rd/'code_snapshot.json'
        cfg=yaml.safe_load(config.read_text()) if config.exists() else {}
        row.update(config_path=rel(config),config_sha256=sha(config) if config.exists() else '',run_code_revision=read(cs).get('code_sha256') if cs.exists() else 'UNKNOWN: no run code_snapshot.json',checkpoint_exists=Path(cp).is_file())
        row['pretrained_weight_version']=json.dumps({k:v for k,v in model_config(cfg).items() if 'weight' in k or 'repo' in k or 'revision' in k},sort_keys=True)
        row['data_contract']=json.dumps(cfg.get('data',{}),sort_keys=True)
        row['training_recipe']=json.dumps(cfg.get('training',{}),sort_keys=True)
        row['model_contract']=json.dumps(model_config(cfg),sort_keys=True)
        summary=read(rd/'run_summary.json') if (rd/'run_summary.json').exists() else {}
        ma=read(rd/'model_audit.json') if (rd/'model_audit.json').exists() else {}
        ga=read(rd/'gradient_audit.json') if (rd/'gradient_audit.json').exists() else {}
        hist=read(rd/'training_history.json') if (rd/'training_history.json').exists() else None
        runtime.append(dict(run_id=r['run_id'],method=method,dataset=r['dataset'],model=r['model'],adaptation=r['adaptation'],seed=r['seed'],status=summary.get('status'),dry_run=summary.get('dry_run'),best_epoch=summary.get('best_epoch'),selection=summary.get('checkpoint_selection'),test_access_during_training_or_selection=summary.get('test_access_during_training_or_selection','not recorded'),trainable={k:v for k,v in ma.items() if 'parameters' in k and not isinstance(v,dict)},gradient_audit=ga,pretrained_load=ma.get('pretrained_load',{}),run_summary_path=rel(rd/'run_summary.json'),history_path=rel(rd/'training_history.json'),history_type=type(hist).__name__,history_length=len(hist) if hist is not None else None,input_artifacts_path=rel(rd/'input_artifacts.json'),run_code_revision=row['run_code_revision']))
    actual.append(row)
csvout('actual_artifacts.csv',actual)
(OUT/'run_evidence.json').write_text(json.dumps(runtime,indent=2))
robust={('eurosat','dofa','frozen'),('treesatai','panopticon','full_finetune'),('cloudsen12','panopticon','frozen'),('spacenet7','dofa','full_finetune')}
coverage=[]
for ds,mo,ad,me in itertools.product(('eurosat','treesatai','cloudsen12','spacenet7'),('dofa','panopticon'),('frozen','full_finetune'),('deterministic','temperature_scaling','mc_dropout','deep_ensemble')):
    records=[r for r in actual if (r['dataset'],r['model'],r['adaptation'],r['uq_method'])==(ds,mo,ad,me)]
    na=me=='temperature_scaling' and ds!='eurosat'
    expected=[] if na else (['ensemble(42,43,44)'] if me=='deep_ensemble' else (['42','43','44'] if me in ('deterministic','temperature_scaling') or (ds,mo,ad) in robust else ['42']))
    observed=[('ensemble('+r['contributing_seeds'].replace(';',',')+')') if me=='deep_ensemble' else r['seed'] for r in records]
    missing=sorted(set(expected)-set(observed))
    coverage.append(dict(dataset=ds,model=mo,adaptation=ad,method=me,declared_seeds=';'.join(expected),actual_seeds=';'.join(observed),declared_result_count=len(expected),actual_result_count=len(records),artifact_count=sum(r['artifact_exists'] for r in records),coverage_status='NA' if na else ('FAIL' if missing or any(not r['artifact_exists'] for r in records) else 'PASS'),missing_reason=('protocol exclusion, not mathematical invalidity of validation reuse' if ds=='treesatai' else 'segmentation TS excluded by frozen protocol') if na else ';'.join(missing),validity_note='Artifact presence only; see 29 checks. MC lacks same-weight dropout-off comparison.' if me=='mc_dropout' else 'Artifact presence only; see 29 checks.',artifact_paths=';'.join(r['artifact_path'] for r in records),protocol_paths='reports/final_training_protocol.md;reports/pre_uq_protocol_freeze.md'))
csvout('coverage_matrix.csv',coverage)
# Direct manifest isolation/group checks; original row IDs are retained in input files.
splits={}; split_rows={}
for ds in ('eurosat','treesatai','cloudsen12','spacenet7'):
    p=ROOT/('splits/eurosat_70_10_10_10_spatial20m/eurosat_splits.csv' if ds=='eurosat' else f'reports/dataset_manifests/{ds}_actual_manifest.csv')
    rr=list(csv.DictReader(p.open())); split_rows[ds]=rr
    groups=collections.defaultdict(list)
    for r in rr:groups[r['split']].append(r)
    keys={'eurosat':['sample_id','sha256','spatial_group'],'treesatai':['sample_id','source_path','advertised_ts_path'],'cloudsen12':['sample_id','roi_id','equi_id','sentinel2_product_id'],'spacenet7':['sample_id','patch_id','aoi','source_image','source_mask']}[ds]
    overlap={}
    for a,b in itertools.combinations(groups,2):
        overlap[a+' vs '+b]={k:sorted(set(r[k] for r in groups[a])&set(r[k] for r in groups[b])) for k in keys}
    splits[ds]={'manifest_path':rel(p),'sha256':sha(p),'counts':{k:len(v) for k,v in groups.items()},'duplicates_within_split':{k:len(v)-len({r['sample_id'] for r in v}) for k,v in groups.items()},'cross_split_overlap':overlap,'final_test_ids':sorted(r['sample_id'] for r in groups['test'])}
(OUT/'split_checks.json').write_text(json.dumps(splits,indent=2))
# Compare configs, disclose differences instead of assuming same LR means fair.
contrasts=[]
for ds,mo,seed in itertools.product(('eurosat','treesatai','cloudsen12','spacenet7'),('dofa','panopticon'),('42','43','44')):
    pair=[next(r for r in actual if r['dataset']==ds and r['model']==mo and r['adaptation']==ad and r['seed']==seed and r['uq_method']=='deterministic') for ad in ('frozen','full_finetune')]
    cfgs=[yaml.safe_load((ROOT/r['config_path']).read_text()) for r in pair]
    contrasts.append({'dataset':ds,'model':mo,'seed':seed,'config_paths':[r['config_path'] for r in pair],'data_equal':cfgs[0].get('data')==cfgs[1].get('data'),'frozen_training':cfgs[0].get('training'),'full_training':cfgs[1].get('training'),'frozen_model':model_config(cfgs[0]),'full_model':model_config(cfgs[1])})
(OUT/'adaptation_config_pairs.json').write_text(json.dumps(contrasts,indent=2))
print(json.dumps({'actual_results':len(actual),'coverage_cells':len(coverage),'coverage_status':dict(collections.Counter(r['coverage_status'] for r in coverage)),'split_counts':{k:v['counts'] for k,v in splits.items()},'missing_artifacts':[r['artifact_path'] for r in actual if not r['artifact_exists']]},indent=2))
