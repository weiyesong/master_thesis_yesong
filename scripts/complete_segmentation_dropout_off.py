"""Evaluate existing C5 weights with all dropout off; never train or overwrite.

One shard per GPU. Complete manifests serve as resume markers. Metric/export
implementations match the frozen project protocol; an independent NumPy audit
is run separately against the exported arrays.
"""
from __future__ import annotations
import argparse, copy, gc, hashlib, json, os, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from scripts.c5_mc_dropout_inference import (_find_training_run, _load_registry, _bn_state, _bn_equal, compact_key)
from scripts.mc_dropout import designated_mc_dropout_modules
from scripts.segmentation_pipeline import (build_segmentation_model, make_segmentation_dataloaders,
    collect_segmentation_predictions, export_segmentation_predictions)
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/final_thesis/core_rq_completion_20260921/segmentation'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=True);f.write('\n')
def evaluate(row,device):
    folder=OUTPUT/row['dataset']/row['model']/row['adaptation']/f"seed{row['seed']}"
    if (folder/'completion.json').exists():
        old=json.loads((folder/'completion.json').read_text())
        assert old['passed'] and sha(folder/'predictions.npz')==old['prediction_sha256']
        print('VERIFIED RESUME',compact_key(row),flush=True);return
    if folder.exists():raise RuntimeError(f'Incomplete output requires inspection, refusing overwrite: {folder}')
    start=time.time();run,summary,config=_find_training_run(row);cp=run/'best.pt';cp_hash=sha(cp)
    original=json.loads((ROOT/'results/final_thesis/mc_dropout/inference/segmentation'/row['dataset']/row['model']/row['adaptation']/f"seed{row['seed']}"/'manifest.json').read_text())
    assert original['checkpoint_sha256']==cp_hash and Path(original['checkpoint_path'])==cp
    code_paths=[Path(__file__).resolve(),ROOT/'scripts/segmentation_pipeline.py',ROOT/'scripts/geobench_datasets.py',ROOT/'scripts/mc_dropout.py']
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in code_paths}
    # Exact model and input configuration, strict full checkpoint restoration.
    exp=next(x for x in config['experiments'] if x['name']==row['experiment'])
    model=build_segmentation_model(config,exp['model']).to(device)
    checkpoint=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    model.eval();bn_before=_bn_state(model)
    modules=designated_mc_dropout_modules(model)
    assert len(modules)==1 and isinstance(modules[0][1],torch.nn.Dropout2d) and modules[0][1].p==.1
    assert not any(m.training for m in model.modules())
    loader=make_segmentation_dataloaders(config)['test'];data=config['data'];metric=config.get('metrics',{})
    print('START',compact_key(row),'N=',len(loader.dataset),'device=',device,flush=True)
    # Pilot checks repeatability on a fixed first batch; no adaptation/fitting.
    raw=next(iter(loader));x=raw['image'].to(device)
    with torch.no_grad():
        a=model(x);b=model(x)
    repeat_error=float((a-b).abs().max().item());assert repeat_error==0
    del raw,x,a,b
    collected=collect_segmentation_predictions(model,loader,device,data['class_names'],data.get('ignore_index'),
        n_bins=15,foreground_class_index=data.get('foreground_class_index'),boundary_radius=int(metric.get('boundary_radius',1)))
    checks={'all_modules_eval':not any(m.training for m in model.modules()),'one_designated_dropout2d_p01_disabled':not modules[0][1].training,
        'batchnorm_buffers_unchanged':_bn_equal(bn_before,_bn_state(model)), 'no_gradients':all(p.grad is None for p in model.parameters()),
        'checkpoint_unchanged':sha(cp)==cp_hash,'same_checkpoint_as_mc':original['checkpoint_sha256']==cp_hash,'evaluation_sources_unchanged':source_hashes=={str(p.relative_to(ROOT)):sha(p) for p in code_paths},'complete_test_count':len(collected['sample_ids'])==len(loader.dataset),
        'unique_ids':len(set(collected['sample_ids']))==len(collected['sample_ids']),'pilot_repeat_logits_exact':repeat_error==0}
    assert all(checks.values()),checks
    source={'run_id':summary['run_id'],'method':'same_checkpoint_dropout_off','checkpoint_sha256':cp_hash,
        'mc_reference_manifest':str(ROOT/'results/final_thesis/mc_dropout/inference/segmentation'/row['dataset']/row['model']/row['adaptation']/f"seed{row['seed']}"/'manifest.json'),
        'resolved_config':str(run/'resolved_config.yaml'),'resolved_config_sha256':sha(run/'resolved_config.yaml'),
        'code_sha256':source_hashes,
        'seed':row['seed'],'training_performed':False,'dropout_off':True,'checkpoint_load_strict':True,
        'evaluation_mode':'full precision eval; no_grad; same test loader; 1 deterministic forward per sample'}
    export_segmentation_predictions(folder,sample_ids=collected['sample_ids'],masks=collected['masks'],logits=collected['logits'],
        class_names=data['class_names'],ignore_index=data.get('ignore_index'),model_name=row['model'],dataset=row['dataset'],
        adaptation_mode=row['adaptation'],split='test',checkpoint=str(cp),representations=collected['representations'],
        per_image_results=collected['per_image_results'],source=source)
    write(folder/'metrics.json',collected['metrics'])
    result={k:row[k] for k in ('task','dataset','model','adaptation','seed')}
    result.update(source);result.update(checks=checks,passed=all(checks.values()),sample_count=len(collected['sample_ids']),
        checkpoint_path=str(cp),prediction_path=str(folder/'predictions.npz'),prediction_sha256=sha(folder/'predictions.npz'),
        metrics_sha256=sha(folder/'metrics.json'),created_at=datetime.now(timezone.utc).isoformat(),duration_seconds=time.time()-start,
        device=str(device),torch_version=torch.__version__,python_version=platform.python_version(),pilot_repeat_max_error=repeat_error)
    write(folder/'completion.json',result)
    print('COMPLETE',compact_key(row),collected['metrics'],'seconds=',result['duration_seconds'],flush=True)
    del collected,loader,model;gc.collect();torch.cuda.empty_cache()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True);parser.add_argument('--shard',type=int,required=True);parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    rows=[r for r in _load_registry()['runs'] if r['task']=='segmentation']
    for i,row in enumerate(rows):
        if i%args.shards==args.shard:evaluate(row,torch.device(args.device))
if __name__=='__main__':main()
