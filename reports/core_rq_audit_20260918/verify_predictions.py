"""Independent NumPy audit, no repository metric function or model execution.

Recomputes all classification outputs and all segmentation outputs in small
image batches. ECE bins are (lower, upper], with zero in the first bin. Values
are fractions, multiclass Brier sums classes; multilabel Brier averages labels.
Writes only to this audit directory; no historical artifact is overwritten.
"""
from pathlib import Path
import csv,json,hashlib,collections,sys,gc
import numpy as np
import pyarrow.parquet as pq
from scipy.special import softmax,expit
from scipy.ndimage import maximum_filter

ROOT=Path('/workspace'); OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'actual_artifacts.csv').open()))
master=list(csv.DictReader((ROOT/'reports/thesis_master_results.csv').open()))
splits=json.loads((OUT/'split_checks.json').read_text())
def key(r):return tuple(r[k] for k in ('dataset','model','adaptation','uq_method','seed'))
master={key(r):r for r in master if r['record_type'] in ('INDIVIDUAL_SEED','ENSEMBLE')}
def writecsv(name,data):
    fields=list(dict.fromkeys(k for r in data for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def fhash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
class Bins:
    def __init__(self,n=15):self.n=n; self.count=np.zeros(n,np.int64);self.conf=np.zeros(n);self.obs=np.zeros(n)
    def update(self,p,y):
        p=np.asarray(p,dtype=np.float64).reshape(-1);y=np.asarray(y).reshape(-1)
        ix=np.clip(np.searchsorted(np.linspace(0,1,self.n+1),p,side='left')-1,0,self.n-1)
        self.count+=np.bincount(ix,minlength=self.n)
        self.conf+=np.bincount(ix,weights=p,minlength=self.n)
        self.obs+=np.bincount(ix,weights=y,minlength=self.n)
    def ece(self):return float(np.abs(self.conf-self.obs).sum()/self.count.sum()) if self.count.sum() else float('nan')
    def records(self,base,kind):
        return [dict(**base,kind=kind,n_bins=self.n,bin=i,lower=i/self.n,upper=(i+1)/self.n,count=int(self.count[i]),mean_confidence=self.conf[i]/self.count[i] if self.count[i] else '',observed_frequency=self.obs[i]/self.count[i] if self.count[i] else '',signed_confidence_minus_outcome=(self.conf[i]-self.obs[i])/self.count[i] if self.count[i] else '') for i in range(self.n)]
def classification(p,y):
    p=p.astype(np.float64);multi=y.ndim==2
    pred=p>=.5 if multi else p.argmax(1)
    if multi:
        q=np.clip(p,np.finfo(float).tiny,1-1e-15)
        out={'accuracy':float((pred==y).all(1).mean()),'nll':float(-(y*np.log(q)+(1-y)*np.log1p(-q)).mean()),'brier':float(((p-y)**2).mean())}
        conf=np.maximum(p,1-p).ravel(); obs=(pred==y).ravel()
    else:
        out={'accuracy':float((pred==y).mean()),'nll':float(-np.log(np.maximum(p[np.arange(len(y)),y],np.finfo(float).tiny)).mean()),'brier':float(((p-np.eye(p.shape[1])[y])**2).sum(1).mean())}
        conf=p.max(1);obs=pred==y
    f1=[]
    for i in range(p.shape[1]):
        yp=y[:,i].astype(bool) if multi else y==i; pp=pred[:,i] if multi else pred==i
        tp=(yp&pp).sum();den=yp.sum()+pp.sum();f1.append(float(2*tp/den) if den else 0)
    out['macro_f1']=float(np.mean(f1));out['mean_confidence_minus_accuracy']=float(np.mean(conf-obs))
    bins={}
    for n in (10,15,30):
        b=Bins(n);b.update(conf,obs);bins[f'decision_{n}']=b;out[f'ece_{n}']=b.ece()
    for i in range(p.shape[1]):
        b=Bins();b.update(p[:,i],y[:,i] if multi else y==i);bins[f'class_probability_{i}']=b
    if multi:
        b=Bins();b.update(p,y);bins['pooled_positive_probability']=b;out['pooled_positive_probability_ece_15']=b.ece()
        out['macro_class_probability_ece_15']=float(np.mean([bins[f'class_probability_{i}'].ece() for i in range(p.shape[1])]))
    return out,bins
def record(r,metrics,checks,bins):
    base={k:r[k] for k in ('dataset','model','adaptation','uq_method','seed')}
    diffs={m:float(v)-float(master[key(r)][m]) for m,v in metrics.items() if m in master[key(r)] and master[key(r)][m]!=''}
    bad={m:v for m,v in diffs.items() if abs(v)>2e-6}
    result=dict(**base,artifact_path=r['artifact_path'],metrics=metrics,metric_differences=diffs,metric_mismatches_over_2e_6=bad,checks=checks)
    for kind,b in bins.items():bin_rows.extend(b.records(base,kind))
    results.append(result)
    with (OUT/(mode+'_verification.json')).open('w') as f:json.dump(results,f,indent=2)
    print('/'.join(key(r)), 'metric_mismatches',bad,'failed_checks',{k:v for k,v in checks.items() if v is False},flush=True)

mode=sys.argv[1] if len(sys.argv)>1 else 'classification'
results=[];bin_rows=[];references={}
for r in rows:
    if r['task']!=mode:continue
    path=ROOT/r['artifact_path'];method=r['uq_method'];ds=r['dataset'];d=path.parent
    if mode=='classification':
        t=pq.read_table(path); names=t.column_names
        p=np.asarray(t['probabilities'].to_pylist()); y=np.asarray(t['label' if 'label' in names else 'true_label'].to_pylist(),dtype=np.int64);ids=np.asarray(t['sample_id'].to_pylist())
        metrics,bins=classification(p,y)
        checks={'full_test_id_set':set(ids)==set(splits[ds]['final_test_ids']),'unique_ids':len(ids)==len(set(ids)),'finite_probabilities':bool(np.isfinite(p).all()),'probability_range':bool(((p>=0)&(p<=1)).all()),'artifact_sha256':fhash(path)}
        order=np.argsort(ids); labels_digest=digest(y[order]);checks['ordered_labels_sha256']=labels_digest
        if ds in references:checks['labels_match_other_methods_models_seeds']=labels_digest==references[ds]
        else:references[ds]=labels_digest
        if method=='deterministic':
            z=np.asarray(t['logits'].to_pylist()); expected=expit(z) if y.ndim==2 else softmax(z,axis=1)
            checks['max_logit_probability_error']=float(np.max(np.abs(p-expected)))
        elif method=='temperature_scaling':
            man=json.loads((d/'metrics_and_manifest.json').read_text());temp=man['fit']['temperature']
            raw=np.asarray(t['raw_logits'].to_pylist());scaled=np.asarray(t['scaled_logits'].to_pylist())
            cal=pq.read_table(man['calibration_prediction_path'],columns=['sample_id']);cids=cal['sample_id'].to_pylist()
            rawsource=pq.read_table(man['test_prediction_path']);srids=rawsource['sample_id'].to_pylist()
            checks.update(positive_scalar_temperature=bool(temp>0 and np.ndim(temp)==0),temperature=temp,scaled_logits_max_error=float(np.max(np.abs(scaled-raw/temp))),argmax_changes=int(np.sum(raw.argmax(1)!=p.argmax(1))),calibration_test_disjoint=not bool(set(cids)&set(ids)),calibration_ids_exact=set(cids)==set(row['sample_id'] for row in csv.DictReader((ROOT/'splits/eurosat_70_10_10_10_spatial20m/eurosat_splits.csv').open()) if row['split']=='calibration'),raw_test_ids_equal=srids==ids.tolist(),raw_logits_equal=bool(np.allclose(np.asarray(rawsource['logits'].to_pylist()),raw,atol=0,rtol=0)))
        elif method=='mc_dropout':
            with np.load(d/'stochastic_outputs.npz',allow_pickle=False) as a:
                sp=a['probabilities'];checks.update(stochastic_shape=list(sp.shape),passes_30=sp.shape[1]==30,stochastic_outputs_differ=bool(np.any(np.ptp(sp,axis=1)>0)),probability_mean_max_error=float(np.max(np.abs(sp.mean(1,dtype=np.float64)-p))),stochastic_ids_equal=bool(np.array_equal(a['sample_ids'],ids)))
        elif method=='deep_ensemble':
            memberfile=next((x for x in (d/'member_predictions.npz',d/'member_outputs.npz') if x.exists()),None)
            man=json.loads((d/'manifest.json').read_text())
            members=man.get('members',[]); checks['member_seeds']=man.get('member_seeds');checks['distinct_member_run_ids']=len({m['run_id'] for m in members})==3 if members else 'see manifest'
            if memberfile:
                with np.load(memberfile,allow_pickle=False) as a:
                    sp=a['probabilities']; checks['member_probability_mean_max_error']=float(np.max(np.abs(sp.mean(axis=1,dtype=np.float64)-p)))
                    checks['member_probability_shape']=list(sp.shape)
        record(r,metrics,checks,bins)
    else:
        # Load only arrays required for metrics. Probability arrays are <1 GB.
        with np.load(path,allow_pickle=False) as a:
            ids=a['sample_id'];y=a['label'];p=a['probabilities'];valid=a['valid_mask'].astype(bool);stored_prediction=a['prediction']
        classes=p.shape[1];confusion=np.zeros((classes,classes),np.int64)
        bins={f'decision_{n}':Bins(n) for n in (10,15,30)}
        bins.update({f'class_probability_{i}':Bins() for i in range(classes)})
        if ds=='spacenet7':bins.update(boundary_top_label=Bins(),boundary_building_probability=Bins())
        sums=collections.defaultdict(float);no_boundary=0;argdiff=0
        for start in range(0,len(y),16):
            yy=y[start:start+16];vv=valid[start:start+16];pr=np.moveaxis(p[start:start+16],1,-1);q=pr[vv].astype(np.float64);lab=yy[vv].astype(int);pred=q.argmax(1);conf=q.max(1)
            argdiff+=int((pred!=stored_prediction[start:start+16][vv]).sum());sums['n']+=len(lab)
            confusion+=np.bincount(classes*lab+pred,minlength=classes**2).reshape(classes,classes)
            sums['nll']+=float(-np.log(np.maximum(q[np.arange(len(lab)),lab],1e-12)).sum())
            sums['brier']+=float(((q-np.eye(classes)[lab])**2).sum())
            sums['gap']+=float((conf-(pred==lab)).sum())
            for n in (10,15,30):bins[f'decision_{n}'].update(conf,pred==lab)
            for c in range(classes):bins[f'class_probability_{c}'].update(q[:,c],lab==c)
            if ds=='spacenet7':
                bq=np.clip(q[:,1],1e-12,1-1e-7);by=lab==1
                sums['fg_nll']+=float(-(by*np.log(bq)+(~by)*np.log1p(-bq)).sum());sums['fg_brier']+=float(((bq-by)**2).sum())
                # Independent implementation of 4-neighbour transitions + radius-1 dilation.
                boundary=np.zeros_like(vv)
                change=(yy[:,1:,:]!=yy[:,:-1,:])&vv[:,1:,:]&vv[:,:-1,:];boundary[:,1:,:]|=change;boundary[:,:-1,:]|=change
                change=(yy[:,:,1:]!=yy[:,:,:-1])&vv[:,:,1:]&vv[:,:,:-1];boundary[:,:,1:]|=change;boundary[:,:,:-1]|=change
                boundary=maximum_filter(boundary,size=(1,3,3),mode='constant')&vv
                no_boundary+=int((~boundary.reshape(len(yy),-1).any(1)).sum());sums['boundary_pixels']+=int(boundary.sum())
                bpr=pr[boundary];bl=yy[boundary];bins['boundary_top_label'].update(bpr.max(1),bpr.argmax(1)==bl);bins['boundary_building_probability'].update(bpr[:,1],bl==1)
        union=confusion.sum(0)+confusion.sum(1)-confusion.diagonal();iou=np.divide(confusion.diagonal(),union,out=np.full(classes,np.nan),where=union>0)
        metrics={'miou':float(np.nanmean(iou)),'pixel_accuracy':float(confusion.trace()/sums['n']),'nll':sums['nll']/sums['n'],'brier':sums['brier']/sums['n'],'mean_confidence_minus_accuracy':sums['gap']/sums['n'],'valid_pixels':int(sums['n']),'confusion_matrix':confusion.tolist()}
        for n in (10,15,30):metrics[f'ece_{n}']=bins[f'decision_{n}'].ece()
        names=['background','building'] if ds=='spacenet7' else ['clear','thick_cloud','thin_cloud','cloud_shadow']
        for i,name in enumerate(names):metrics[f'iou_{name}']=float(iou[i]);metrics[f'class_probability_ece_{name}']=bins[f'class_probability_{i}'].ece()
        if ds=='spacenet7':metrics.update(foreground_ece_15=bins['class_probability_1'].ece(),foreground_nll=sums['fg_nll']/sums['n'],foreground_brier=sums['fg_brier']/sums['n'],boundary_ece_15=bins['boundary_top_label'].ece(),boundary_foreground_ece_15=bins['boundary_building_probability'].ece(),boundary_pixels=int(sums['boundary_pixels']),images_without_boundary=no_boundary)
        order=np.argsort(ids);ld=digest(y[order]);vd=digest(valid[order])
        checks={'full_test_id_set':set(ids)==set(splits[ds]['final_test_ids']),'unique_ids':len(ids)==len(set(ids)),'ordered_labels_sha256':ld,'ordered_valid_mask_sha256':vd,'argmax_vs_saved_prediction_pixel_difference':argdiff,'mask_matches_ignore_contract':bool(np.array_equal(valid,y!=255)),'finite_probabilities':bool(np.isfinite(p).all()),'probability_sum_max_error':float(np.abs(p.sum(1)-1).max()),'artifact_sha256':fhash(path)}
        if ds in references:checks['labels_and_mask_match_other_methods_models_seeds']=(ld,vd)==references[ds]
        else:references[ds]=(ld,vd)
        record(r,metrics,checks,bins)
        del y,p,valid,stored_prediction;gc.collect()
    writecsv(mode+'_reliability_bins.csv',bin_rows)
flat=[]
for r in results:flat.append({k:v for k,v in r.items() if k not in ('metrics','checks','metric_differences','metric_mismatches_over_2e_6')}|{k:v for k,v in r['metrics'].items() if not isinstance(v,list)})
writecsv(mode+'_recomputed_metrics.csv',flat)
print('DONE',mode,len(results),flush=True)
