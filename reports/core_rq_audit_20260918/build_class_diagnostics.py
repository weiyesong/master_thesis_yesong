"""Direct label-probability and per-class performance audit tables."""
from pathlib import Path
import csv,json
import numpy as np
import pyarrow.parquet as pq
ROOT=Path('/workspace');OUT=Path(__file__).resolve().parent
def csvout(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ece(p,y):
 p=p.astype(np.float64);ix=np.clip(np.searchsorted(np.linspace(0,1,16),p,side='left')-1,0,14)
 cp=np.bincount(ix,weights=p,minlength=15);cy=np.bincount(ix,weights=y,minlength=15)
 return float(np.abs(cp-cy).sum()/len(p))
names={'treesatai':['Abies','Acer','Alnus','Betula','Cleared','Fagus','Fraxinus','Larix','Picea','Pinus','Populus','Prunus','Pseudotsuga','Quercus','Tilia'],'eurosat':['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']}
out=[]
for r in csv.DictReader((OUT/'actual_artifacts.csv').open()):
 if r['task']!='classification':continue
 t=pq.read_table(ROOT/r['artifact_path']);y=np.asarray(t['label' if 'label' in t.column_names else 'true_label'].to_pylist());p=np.asarray(t['probabilities'].to_pylist());pred=p>=.5 if y.ndim==2 else p.argmax(1)
 for c in range(p.shape[1]):
  label=y[:,c].astype(bool) if y.ndim==2 else y==c;decision=pred[:,c] if y.ndim==2 else pred==c;tp=int((label&decision).sum());fp=int((~label&decision).sum());fn=int((label&~decision).sum());tn=int((~label&~decision).sum());prob=p[:,c]
  row={k:r[k] for k in ('dataset','model','adaptation','uq_method','seed')}
  row.update(class_index=c,class_name=names[r['dataset']][c],samples=len(y),support=int(label.sum()),predicted_positive=int(decision.sum()),tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,class_probability_ece_15=ece(prob,label),mean_probability=float(prob.mean()),prevalence=float(label.mean()),mean_probability_on_true_positive_labels=float(prob[label].mean()) if label.any() else '',artifact_path=r['artifact_path'])
  out.append(row)
csvout('class_diagnostics.csv',out)
out=[]
for r in json.loads((OUT/'segmentation_verification.json').read_text()):
 cm=np.array(r['metrics']['confusion_matrix']);ns=['background','building'] if r['dataset']=='spacenet7' else ['clear','thick_cloud','thin_cloud','cloud_shadow']
 for i,name in enumerate(ns):
  tp=int(cm[i,i]);support=int(cm[i].sum());predicted=int(cm[:,i].sum());row={k:r[k] for k in ('dataset','model','adaptation','uq_method','seed')};row.update(class_index=i,class_name=name,valid_pixels=int(cm.sum()),support=support,predicted_positive=predicted,tp=tp,fp=predicted-tp,fn=support-tp,precision=tp/predicted if predicted else 0,recall=tp/support if support else 0,iou=r['metrics']['iou_'+name],class_probability_ece_15=r['metrics']['class_probability_ece_'+name],artifact_path=r['artifact_path']);out.append(row)
csvout('segmentation_class_diagnostics.csv',out)
print('segmentation class rows',len(out))
