"""Check aggregation from direct members; compressed-NPY slice reads bound cost."""
from pathlib import Path
import csv,json,zipfile
import numpy as np
import pyarrow.parquet as pq
ROOT=Path('/workspace');OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'actual_artifacts.csv').open()))
def npyslice(path,name,start=0,count=8):
 with zipfile.ZipFile(path) as z, z.open(name+'.npy') as f:
  version=np.lib.format.read_magic(f)
  shape,fortran,dtype=np.lib.format._read_array_header(f,version)
  if fortran or dtype.hasobject:raise ValueError('Unexpected array layout')
  stride=int(np.prod(shape[1:]))*dtype.itemsize
  f.seek(start*stride,1)
  return np.frombuffer(f.read(min(count,shape[0]-start)*stride),dtype=dtype).reshape((-1,*shape[1:]))
results=[]
for r in rows:
 if r['uq_method']!='deep_ensemble':continue
 p=ROOT/r['artifact_path']; members=[s for s in rows if s['dataset']==r['dataset'] and s['model']==r['model'] and s['adaptation']==r['adaptation'] and s['uq_method']=='deterministic'];members=sorted(members,key=lambda s:s['seed'])
 arrays=[];ids=[];labels=[]
 for s in members+[r]:
  sp=ROOT/s['artifact_path']
  if r['task']=='classification':
   t=pq.read_table(sp);names=t.column_names; arrays.append(np.asarray(t['probabilities'].to_pylist()));ids.append(t['sample_id'].to_pylist());labels.append(np.asarray(t['label' if 'label' in names else 'true_label'].to_pylist()))
  else:
   arrays.append(npyslice(sp,'probabilities'));labels.append(npyslice(sp,'label'))
   with np.load(sp,allow_pickle=False) as a:ids.append(a['sample_id'].tolist())
 mean=np.mean(np.stack(arrays[:3],axis=1).astype(np.float64),axis=1)
 results.append(dict(dataset=r['dataset'],model=r['model'],adaptation=r['adaptation'],method='deep_ensemble',seeds=[s['seed'] for s in members],distinct_member_runs=len({s['run_id'] for s in members})==3,independent_checkpoint_hashes=len({s['checkpoint_sha256'] for s in members})==3,all_sample_ids_exactly_aligned=all(i==ids[-1] for i in ids),labels_exactly_aligned=all(np.array_equal(l,labels[-1]) for l in labels),sample_scope='all images' if r['task']=='classification' else 'first 8 images in artifact order, full spatial maps',probability_mean_max_absolute_error=float(np.max(np.abs(mean-arrays[-1]))),member_artifact_paths=[s['artifact_path'] for s in members],ensemble_artifact_path=r['artifact_path']))
 print(results[-1],flush=True)
 (OUT/'aggregation_verification.json').write_text(json.dumps(results,indent=2))
for r in rows:
 if r['uq_method']!='mc_dropout' or r['task']!='segmentation':continue
 p=ROOT/r['artifact_path']; sub=p.parent/'research_subset_stochastic_probabilities.npz'
 with np.load(sub,allow_pickle=False) as a:ids=a['sample_id'].tolist()
 with np.load(p,allow_pickle=False) as a:allids=a['sample_id'].tolist()
 # First prospectively hash-selected research-subset image, not selected by result.
 passes=npyslice(sub,'probabilities',count=1);index=allids.index(ids[0]);mean=npyslice(p,'probabilities',start=index,count=1)
 results.append(dict(dataset=r['dataset'],model=r['model'],adaptation=r['adaptation'],method='mc_dropout',seed=r['seed'],sample_id=ids[0],sample_scope='first predeclared research-subset image; all 30 passes and every pixel',passes=passes.shape[1],stochastic_outputs_differ=bool(np.any(np.ptp(passes,axis=1)>0)),probability_mean_max_absolute_error=float(np.max(np.abs(passes.astype(np.float64).mean(1)-mean))),labels_equal=bool(np.array_equal(npyslice(sub,'label',count=1),npyslice(p,'label',index,count=1))),stochastic_artifact_path=str(sub.relative_to(ROOT)),aggregate_artifact_path=r['artifact_path']))
 print(results[-1],flush=True)
 (OUT/'aggregation_verification.json').write_text(json.dumps(results,indent=2))
print('DONE',len(results),flush=True)
