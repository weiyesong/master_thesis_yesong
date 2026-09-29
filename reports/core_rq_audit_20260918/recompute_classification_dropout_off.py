"""Same-checkpoint dropout-off from saved deterministic backbone features.

No backbone inference or training. New outputs are confined to this audit.
"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq
torch.set_num_threads(2)
ROOT=Path('/workspace');OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'actual_artifacts.csv').open()))
verified=json.loads((OUT/'classification_verification.json').read_text())
def metric(p,y):
 p=np.asarray(p,dtype=np.float64);multi=y.ndim==2
 if multi:
  pred=p>=.5; accuracy=(pred==y).all(1).mean(); q=np.clip(p,np.finfo(float).tiny,1-1e-15)
  nll=-(y*np.log(q)+(1-y)*np.log1p(-q)).mean();brier=((p-y)**2).mean();conf=np.maximum(p,1-p).ravel();correct=(pred==y).ravel()
 else:
  pred=p.argmax(1);accuracy=(pred==y).mean();nll=-np.log(np.maximum(p[np.arange(len(y)),y],np.finfo(float).tiny)).mean();brier=((p-np.eye(p.shape[1])[y])**2).sum(1).mean();conf=p.max(1);correct=pred==y
 f1=[]
 for c in range(p.shape[1]):
  yp=y[:,c].astype(bool) if multi else y==c;pp=pred[:,c] if multi else pred==c;den=yp.sum()+pp.sum();f1.append(2*(yp&pp).sum()/den if den else 0)
 edges=np.linspace(0,1,16);ix=np.clip(np.searchsorted(edges,conf,side='left')-1,0,14);count=np.bincount(ix,minlength=15);cp=np.bincount(ix,weights=conf,minlength=15);cy=np.bincount(ix,weights=correct,minlength=15)
 return dict(accuracy=float(accuracy),macro_f1=float(np.mean(f1)),nll=float(nll),brier=float(brier),ece_15=float(np.abs(cp-cy).sum()/count.sum()))
results=[]
for r in rows:
 if r['uq_method']!='mc_dropout' or r['task']!='classification':continue
 artifact=ROOT/r['artifact_path'];t=pq.read_table(artifact,columns=['sample_id','label','backbone_representation']);features=np.asarray(t['backbone_representation'].to_pylist(),dtype=np.float32);y=np.asarray(t['label'].to_pylist(),dtype=np.int64);ids=t['sample_id'].to_pylist()
 with np.load(artifact.parent/'embeddings.npz',allow_pickle=False) as a:
  assert np.array_equal(a['embeddings'],features);assert a['sample_ids'].tolist()==ids
 checkpoint=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=False,mmap=True);state=checkpoint['model'];hc=checkpoint['experiment']['model']['head'];assert hc['dropout']==.1
 x=torch.from_numpy(features)
 if hc['architecture']=='batchnorm_linear':
  assert not hc['batchnorm_affine'];x=F.batch_norm(x,state['head.0.running_mean'],state['head.0.running_var'],training=False,eps=hc['batchnorm_eps']);layer='head.2'
 elif hc['architecture']=='linear':layer='head.1'
 else:raise ValueError('Unreviewed head architecture')
 logits=F.linear(x,state[layer+'.weight'],state[layer+'.bias']);p=logits.sigmoid() if y.ndim==2 else logits.softmax(1);p=p.numpy();m=metric(p,y)
 lookup=lambda method:next(x for x in verified if all(x[k]==r[k] for k in ('dataset','model','adaptation','seed')) and x['uq_method']==method)
 mc=lookup('mc_dropout');det=lookup('deterministic');folder=OUT/'classification_dropout_off'/r['dataset']/r['model']/r['adaptation']/('seed'+r['seed']);folder.mkdir(parents=True,exist_ok=True)
 pq.write_table(pa.table({'sample_id':ids,'label':y.tolist(),'logits':logits.numpy().tolist(),'probabilities':p.tolist()}),folder/'predictions.parquet')
 res={k:r[k] for k in ('dataset','model','adaptation','seed','run_id','checkpoint_path','checkpoint_sha256')};res.update(feature_artifact_path=r['artifact_path'],output_path=str((folder/'predictions.parquet').relative_to(ROOT)),method='same_checkpoint_dropout_off',evaluation='CPU saved backbone representations + eval BatchNorm where present + final Linear; dropout disabled; no backbone inference',same_checkpoint_as_mc=True,feature_ids_and_values_match_saved_embeddings=True,metrics=m,mc_metrics={k:mc['metrics'][k] for k in m},original_p0_metrics={k:det['metrics'][k] for k in m},delta_mc_minus_same_weight_off={k:mc['metrics'][k]-m[k] for k in m},delta_off_minus_original_p0={k:m[k]-det['metrics'][k] for k in m},delta_mc_minus_original_p0={k:mc['metrics'][k]-det['metrics'][k] for k in m})
 (folder/'metrics_and_provenance.json').write_text(json.dumps(res,indent=2));results.append(res)
 print(r['dataset'],r['model'],r['adaptation'],r['seed'],m,'MC-OFF',res['delta_mc_minus_same_weight_off'],flush=True)
(OUT/'classification_dropout_off_verification.json').write_text(json.dumps(results,indent=2))
flat=[]
for r in results:
 z={k:r[k] for k in ('dataset','model','adaptation','seed','checkpoint_path','checkpoint_sha256','feature_artifact_path','output_path')}
 for prefix in ('metrics','mc_metrics','original_p0_metrics','delta_mc_minus_same_weight_off','delta_off_minus_original_p0','delta_mc_minus_original_p0'):
  z.update({prefix+'_'+k:v for k,v in r[prefix].items()})
 flat.append(z)
with (OUT/'classification_dropout_off_three_way.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print('DONE',len(results),flush=True)
