"""Align one supplemental metric bundle with its saved CPU prediction export.

Preserves the original GPU accumulation and completion manifest unchanged.
No model is loaded and no inference/training/label modification is performed.
"""
from pathlib import Path
import csv,hashlib,json,sys
from datetime import datetime,timezone
import numpy as np
import torch
ROOT=Path('/workspace');sys.path.insert(0,str(ROOT))
from scripts.segmentation_pipeline import SegmentationMetricAccumulator,segmentation_metrics
DIRECTORY=ROOT/'results/final_thesis/core_rq_completion_20260921/segmentation/spacenet7/panopticon/full_finetune/seed42'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def write(p,obj):
 with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=True);f.write('\n')
def main():
 torch.set_num_threads(2)
 if (DIRECTORY/'metric_alignment.json').exists():
  x=json.loads((DIRECTORY/'metric_alignment.json').read_text())
  assert all(sha(DIRECTORY/name)==digest for name,digest in x['artifact_hashes'].items())
  print('Verified existing alignment, no changes.');return
 files=['predictions.npz','metrics.json','completion.json','per_image_metrics.csv']
 before={f:sha(DIRECTORY/f) for f in files}
 with np.load(DIRECTORY/'predictions.npz',allow_pickle=False) as a:
  ids=a['sample_id'];y=a['label'];z=a['logits'];p=a['probabilities'];pred=a['prediction']
 acc=SegmentationMetricAccumulator(2,['background','building'],255,15,1,1);rows=[];maxerr=0.;argdiff=0
 with torch.no_grad():
  for start in range(0,len(ids),8):
   logits=torch.from_numpy(z[start:start+8]);target=torch.from_numpy(y[start:start+8]);q=logits.softmax(1)
   maxerr=max(maxerr,float(np.abs(q.numpy()-p[start:start+8]).max()));argdiff+=int((q.argmax(1).numpy()!=pred[start:start+8]).sum())
   acc.update(logits,target)
   for i in range(len(logits)):
    m=segmentation_metrics(logits[i:i+1],target[i:i+1],['background','building'],255,15,1,1)
    r={'sample_id':str(ids[start+i]),**{k:m[k] for k in ('miou','pixel_accuracy','nll','brier','ece_15','valid_pixels','ignored_pixels')}}
    r.update({'iou_'+k:v for k,v in m['per_class_iou'].items()})
    r.update({k:m[k] for k in ('foreground_ece_15','foreground_nll','foreground_brier')})
    r.update({'boundary_'+k:v for k,v in m['boundary_calibration'].items() if k in ('ece_15','foreground_ece_15')});rows.append(r)
 assert maxerr==0 and argdiff==0,(maxerr,argdiff)
 canonical=acc.compute();write(DIRECTORY/'metrics_export_aligned.json',canonical)
 with (DIRECTORY/'per_image_metrics_export_aligned.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 assert before=={f:sha(DIRECTORY/f) for f in files}
 original=json.loads((DIRECTORY/'metrics.json').read_text())
 evidence={'created_at':datetime.now(timezone.utc).isoformat(),'reason':'GPU collection softmax versus CPU export softmax at a near-tie pixel; align reporting exactly with stored predictions. Original records preserved.',
  'canonical_metrics_file':'metrics_export_aligned.json','canonical_per_image_file':'per_image_metrics_export_aligned.csv',
  'original_files_unchanged':True,'no_inference_or_training':True,'cpu_recomputed_softmax_max_export_error':maxerr,'cpu_recomputed_argmax_export_difference':argdiff,
  'original_confusion_matrix':original['confusion_matrix'],'canonical_confusion_matrix':canonical['confusion_matrix'],
  'scalar_metric_changes':{k:canonical[k]-original[k] for k in ('miou','pixel_accuracy','ece_15','nll','brier','foreground_nll','foreground_brier','foreground_ece_15')},
  'artifact_hashes':before|{f:sha(DIRECTORY/f) for f in ('metrics_export_aligned.json','per_image_metrics_export_aligned.csv')},
  'code_sha256':{str(f.relative_to(ROOT)):sha(f) for f in (Path(__file__).resolve(),ROOT/'scripts/segmentation_pipeline.py')}}
 write(DIRECTORY/'metric_alignment.json',evidence);print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
