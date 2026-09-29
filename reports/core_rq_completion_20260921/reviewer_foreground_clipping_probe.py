"""Reproduce the reviewer's alternate clipping formula versus project contract."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path('/workspace');OUT=Path(__file__).resolve().parent
D=ROOT/'results/final_thesis/core_rq_completion_20260921/segmentation/spacenet7/panopticon/full_finetune/seed42'
with np.load(D/'predictions.npz',allow_pickle=False) as a:p=a['probabilities'];y=a['label'];valid=a['valid_mask']
protocol=0.;alternate=0.;n=0
for i in range(0,len(y),8):
 v=valid[i:i+8];b=p[i:i+8,1][v];isbuilding=y[i:i+8][v]==1
 q=np.clip(b,np.float32(1e-12),np.float32(1-1e-7))
 protocol+=float((-(isbuilding*np.log(q)+(~isbuilding)*np.log(1-q))).sum(dtype=np.float64))
 true_probability=np.where(isbuilding,b,1-b);true_probability=np.clip(true_probability,np.float32(1e-7),np.float32(1-1e-7))
 alternate+=float((-np.log(true_probability)).sum(dtype=np.float64));n+=len(b)
m=json.loads((D/'metrics_export_aligned.json').read_text())
out={'artifact_path':str((D/'predictions.npz').relative_to(ROOT)),'valid_pixels':n,'protocol':'clip p(building) to float32 [1e-12, 1-1e-7], then binary log likelihood over all valid pixels',
 'reviewer_alternative':'select true-class probability first, then clip to float32 [1e-7, 1-1e-7]; different lower bound and clipping position',
 'independent_protocol_foreground_nll':protocol/n,'reported_canonical_foreground_nll':m['foreground_nll'],'difference_from_reported':protocol/n-m['foreground_nll'],
 'reviewer_alternative_foreground_nll':alternate/n,'alternative_minus_protocol':(alternate-protocol)/n,'passed_protocol_reproduction':abs(protocol/n-m['foreground_nll'])<2e-6}
assert out['passed_protocol_reproduction'];(OUT/'reviewer_foreground_clipping_probe.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
