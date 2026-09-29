"""Resolve foreground-NLL differences due to the saved pipeline dtype/clip.

Computes the formula directly; does not call project metric code. This result
supplements, and does not erase, the float64 raw-probability reanalysis.
"""
import csv,json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path('/workspace');OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'actual_artifacts.csv').open()))
master=list(csv.DictReader((ROOT/'reports/thesis_master_results.csv').open()))
out=[]
for r in rows:
 if r['dataset']!='spacenet7':continue
 with np.load(ROOT/r['artifact_path'],allow_pickle=False) as a:p=a['probabilities'];y=a['label'];mask=a['valid_mask'].astype(bool)
 total=0.;n=0
 for start in range(0,len(y),8):
  q=torch.from_numpy(p[start:start+8])
  if r['uq_method']=='deep_ensemble':q=q.float().clamp_min(1e-30).log().softmax(1)
  elif r['uq_method']=='mc_dropout':q=q.clamp_min(1e-12).log().softmax(1)
  # Match float32 versus float64 clamp representability, retain double accumulation.
  bq=q[:,1].numpy();bq=np.clip(bq,1e-12,1-1e-7);by=y[start:start+8]==1;valid=mask[start:start+8]
  total+=float((-(by*np.log(bq)+(~by)*np.log(1-bq)))[valid].sum(dtype=np.float64));n+=int(valid.sum())
 row=next(m for m in master if all(m[k]==r[k] for k in ('dataset','model','adaptation','uq_method','seed')) and m['record_type']==r['record_type'])
 result={k:r[k] for k in ('dataset','model','adaptation','uq_method','seed','artifact_path')};result.update(saved_probability_dtype=str(p.dtype),reproduced_native_foreground_nll=total/n,reported_foreground_nll=float(row['foreground_nll']),difference=total/n-float(row['foreground_nll']))
 out.append(result);(OUT/'foreground_precision_verification.json').write_text(json.dumps(out,indent=2));print(result,flush=True)
print('DONE',len(out),flush=True)
