"""Read saved tensors and histories; no model construction or inference."""
from pathlib import Path
import csv,json,hashlib
import torch
torch.set_num_threads(2)
ROOT=Path('/workspace');OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'actual_artifacts.csv').open()))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
paths={'dofa':ROOT/'DOFA/checkpoints/DOFA_ViT_base_e100.pth','panopticon':ROOT/'models/pretrained_cache/hub/checkpoints/panopticon_vitb14_teacher.pth'}
pre={k:torch.load(p,map_location='cpu',weights_only=False,mmap=True) for k,p in paths.items()}
out=[]
for r in rows:
 if r['uq_method'] not in ('deterministic','mc_dropout'):continue
 p=Path(r['checkpoint_path']);rd=p.parent
 ck=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
 state=ck['model'];summary=json.loads((rd/'run_summary.json').read_text());hist=json.loads((rd/'training_history.json').read_text())
 metric='nll' if r['task']=='classification' else 'miou';factor=1 if metric=='nll' else -1
 best=min(hist,key=lambda h:(factor*float(h['val'][metric]),int(h['epoch'])))
 prefix=('backbone.' if r['task']=='classification' else 'backbone.backbone.')+('model.' if r['model']=='panopticon' else '')
 equal=[];different=[];unmatched=[]
 for k,t in state.items():
  if not k.startswith(prefix):continue
  name=k[len(prefix):]
  if name not in pre[r['model']] or pre[r['model']][name].shape!=t.shape:unmatched.append(name);continue
  (equal if torch.equal(t,pre[r['model']][name]) else different).append(name)
 out.append(dict(dataset=r['dataset'],model=r['model'],adaptation=r['adaptation'],method=r['uq_method'],seed=r['seed'],run_id=r['run_id'],checkpoint_path=str(p),checkpoint_sha256_actual=sha(p),checkpoint_sha256_expected=r['checkpoint_sha256'],checkpoint_hash_matches=sha(p)==r['checkpoint_sha256'],checkpoint_epoch=ck['epoch'],history_selected_epoch=best['epoch'],summary_selected_epoch=summary['best_epoch'],earliest_validation_selection_correct=ck['epoch']==best['epoch']==summary['best_epoch'],selection_metric=metric,checkpoint_run_id_matches=ck['run_id']==r['run_id'],checkpoint_config_keys=list(ck['config']),pretrained_path=str(paths[r['model']]),equal_backbone_tensor_count=len(equal),changed_backbone_tensor_count=len(different),changed_backbone_tensors=different,unmatched_backbone_tensors=unmatched,frozen_unchanged_or_full_changed=(not different and bool(equal)) if r['adaptation']=='frozen' else bool(different),run_summary_path=str(rd/'run_summary.json'),history_path=str(rd/'training_history.json')))
 (OUT/'checkpoint_verification.json').write_text(json.dumps(out,indent=2))
 print(r['run_id'],out[-1]['checkpoint_hash_matches'],out[-1]['earliest_validation_selection_correct'],'changed',len(different),'unmatched',unmatched,flush=True)
print('DONE',len(out),flush=True)
