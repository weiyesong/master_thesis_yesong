"""Direct CPU/export versus CUDA softmax replay from saved logits; no model inference."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace');OUT=ROOT/'reports/core_rq_completion_20260921'
p=ROOT/'results/final_thesis/core_rq_completion_20260921/segmentation/spacenet7/panopticon/full_finetune/seed42/predictions.npz'
torch.set_num_threads(2)
with np.load(p,allow_pickle=False) as a:
 ids=a['sample_id'];z=a['logits'];p_cpu=a['probabilities'];y=a['label'];valid=a['valid_mask'].astype(bool);saved=a['prediction']
cm_cpu=np.zeros((2,2),np.int64);cm_gpu=np.zeros((2,2),np.int64);points=[];maxprob=0.;maxcpureplay=0.
for start in range(0,len(ids),8):
 zz=torch.from_numpy(z[start:start+8]);cpu=zz.softmax(1).numpy();gpu=zz.cuda(0).softmax(1).cpu().numpy();probs=p_cpu[start:start+8];mask=valid[start:start+8];lab=y[start:start+8]
 maxprob=max(maxprob,float(np.abs(cpu-gpu).max()));maxcpureplay=max(maxcpureplay,float(np.abs(cpu-probs).max()))
 apcpu=probs.argmax(1);apgpu=gpu.argmax(1)
 cm_cpu+=np.bincount((2*lab[mask]+apcpu[mask]).astype(int),minlength=4).reshape(2,2)
 cm_gpu+=np.bincount((2*lab[mask]+apgpu[mask]).astype(int),minlength=4).reshape(2,2)
 for i,h,w in np.argwhere((apcpu!=apgpu)&mask):
  points.append(dict(sample_index=int(start+i),sample_id=str(ids[start+i]),row=int(h),column=int(w),label=int(lab[i,h,w]),logits=z[start+i,:,h,w].tolist(),logit_difference_building_minus_background=float(z[start+i,1,h,w]-z[start+i,0,h,w]),exported_cpu_probabilities=probs[i,:,h,w].tolist(),replayed_cpu_probabilities=cpu[i,:,h,w].tolist(),replayed_cuda_probabilities=gpu[i,:,h,w].tolist(),exported_prediction=int(saved[start+i,h,w]),cuda_prediction=int(apgpu[i,h,w])))
original=json.loads((p.parent/'metrics.json').read_text())
result=dict(artifact_path=str(p.relative_to(ROOT)),torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),dtype=str(z.dtype),replay_batch_size=8,full_test_images=len(ids),valid_pixels=int(valid.sum()),max_cpu_replay_vs_export_probability_error=maxcpureplay,max_cpu_cuda_probability_error=maxprob,argmax_difference_count=len(points),difference_pixels=points,exported_cpu_confusion_matrix=cm_cpu.tolist(),replayed_cuda_confusion_matrix=cm_gpu.tolist(),original_metrics_confusion_matrix=original['confusion_matrix'],cuda_replay_matches_original_confusion_matrix=bool(np.array_equal(cm_gpu,original['confusion_matrix'])),exported_cpu_replay_exact=maxcpureplay==0,scope='Replay of existing logits only. No model inference or training. Original metric and prediction files unchanged.')
(OUT/'segmentation_off_softmax_alignment_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
