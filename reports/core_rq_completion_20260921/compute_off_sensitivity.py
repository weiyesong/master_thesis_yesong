"""Prespecified ECE10/15/30 sensitivity for all matched MC-Off contrasts."""
from pathlib import Path
import csv,json
import numpy as np
import pyarrow.parquet as pq
ROOT=Path('/workspace');OUT=Path(__file__).resolve().parent;OLD=ROOT/'reports/core_rq_audit_20260918'
def ece(p,y,n):
 if y.ndim==2:conf=np.maximum(p,1-p).ravel();obs=((p>=.5)==y).ravel()
 else:conf=p.max(1);obs=p.argmax(1)==y
 ix=np.clip(np.searchsorted(np.linspace(0,1,n+1),conf,side='left')-1,0,n-1)
 return float(np.abs(np.bincount(ix,weights=conf,minlength=n)-np.bincount(ix,weights=obs,minlength=n)).sum()/len(conf))
def key(r):return tuple(str(r[k]) for k in ('dataset','model','adaptation','seed'))
def main():
 comparisons=list(csv.DictReader((OUT/'mc_dropout_three_way.csv').open()));seg={key(r):r for r in json.loads((OUT/'segmentation_off_verification.json').read_text())}
 oldseg={(key(r),r['uq_method']):r for r in json.loads((OLD/'segmentation_verification.json').read_text())}
 rows=[]
 for r in comparisons:
  record={k:r[k] for k in ('dataset','model','adaptation','seed','output_path','mc_path')}
  if r['task']=='classification':
   off=pq.read_table(ROOT/r['output_path']);mc=pq.read_table(ROOT/r['mc_path']);a=np.asarray(off['probabilities'].to_pylist());b=np.asarray(mc['probabilities'].to_pylist());y=np.asarray(off['label'].to_pylist())
   assert off['sample_id'].to_pylist()==mc['sample_id'].to_pylist() and np.array_equal(y,np.asarray(mc['label'].to_pylist()))
   for n in (10,15,30):record[f'Off_ece_{n}']=ece(a,y,n);record[f'MC_ece_{n}']=ece(b,y,n)
  else:
   v=seg[key(r)];assert v['status']=='PASS'
   for n in (10,15,30):record[f'Off_ece_{n}']=v['metrics'][f'ece_{n}'];record[f'MC_ece_{n}']=oldseg[(key(r),'mc_dropout')]['metrics'][f'ece_{n}']
  for n in (10,15,30):record[f'delta_ece_{n}']=record[f'MC_ece_{n}']-record[f'Off_ece_{n}']
  record['sign_consistent_10_15_30']=len({np.sign(record[f'delta_ece_{n}']) for n in (10,15,30)})==1
  record['primary_remains_ece15']=True;rows.append(record)
 with (OUT/'mc_dropout_ece_sensitivity.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('24 sensitivity rows; changing signs:',[(key(r),[r[f'delta_ece_{n}'] for n in (10,15,30)]) for r in rows if not r['sign_consistent_10_15_30']])
if __name__=='__main__':main()
