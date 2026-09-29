"""A: saved-prediction analyses only. Never imports a model or Torch."""
from __future__ import annotations
import argparse, csv, gc, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.special import logsumexp, expit, softmax
from scipy.stats import spearmanr
from scipy.ndimage import maximum_filter
from common import (ROOT,OUT,PARAMS,ARCHIVE,MASTER,get_inputs,expected_row,archive_path,
                    LazyNPZ,dump,csvout,arrhash,groups_for,rss_gib,matrix_col)
from metrics import RankPlan,base_scores,decomposition,entropy,align_indices,interval,bootstrap_group_counts,COVERAGES

BOOT=PARAMS['bootstrap_replicates']
CURVE_Q=np.arange(1,101)/100


def correlation(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if np.ptp(x)==0 or np.ptp(y)==0:return np.nan
    return float(spearmanr(x,y).statistic)


def read_classification(r):
    path=ROOT/r['prediction_path'];z=None;raw_z=None;extra={};binary=r['dataset']=='treesatai'
    if path.suffix=='.parquet':
        available=pq.read_schema(path).names
        wanted=[n for n in ['sample_id','label','true_label','probabilities','logits','scaled_logits','raw_logits','temperature'] if n in available]
        d=pq.read_table(path,columns=wanted).to_pandas();ids=d.sample_id.to_numpy().astype(str)
        labelcol='label' if 'label' in d else 'true_label';y=matrix_col(d,labelcol) if binary else d[labelcol].to_numpy()
        p=matrix_col(d,'probabilities').astype(np.float64)
        if 'logits' in d:z=matrix_col(d,'logits').astype(np.float64)
        if 'scaled_logits' in d:z=matrix_col(d,'scaled_logits').astype(np.float64);raw_z=matrix_col(d,'raw_logits').astype(np.float64);extra['temperature']=float(d.temperature.iloc[0])
    else:
        with np.load(path,allow_pickle=True) as d:
            ids=d['sample_id' if 'sample_id' in d else 'sample_ids'].astype(str)
            y=d['label' if 'label' in d else 'true_labels']
            p=np.asarray(d['mean_probabilities' if 'mean_probabilities' in d else 'probabilities'],dtype=np.float64)
    order=np.argsort(ids);ids=ids[order];p=p[order];y=y[order]
    if z is not None:z=z[order]
    if raw_z is not None:raw_z=raw_z[order]
    if not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('invalid probabilities')
    if not binary and np.max(abs(p.sum(axis=-1)-1))>2e-6:raise ValueError('probabilities not normalized')
    scores=base_scores(p,binary);validation=[]
    if r['raw_path']:
        with np.load(ROOT/r['raw_path'],allow_pickle=True) as raw:
            raw_ids=raw['sample_id' if 'sample_id' in raw else 'sample_ids'].astype(str)
            raw_order=align_indices(raw_ids,ids)
            draws=raw['probabilities'][raw_order]
            raw_y=raw['true_labels' if 'true_labels' in raw else 'label'][raw_order]
            if not np.array_equal(raw_y,y):raise ValueError('raw labels differ')
            mean,dec=decomposition(draws,binary)
            diff=float(np.max(abs(mean-p)))
            if diff>2e-6:raise ValueError(('mean probabilities mismatch',diff))
            residual=float(np.max(abs(dec['gini_total']-dec['gini_expected']-dec['gini_disagreement'])))
            assert residual<2e-12
            validation.append({'check':'raw_mean_vs_saved','max_abs_error':diff,'tolerance':2e-6,'status':'PASS'})
            validation.append({'check':'raw_gini_identity','max_abs_error':residual,'tolerance':2e-12,'status':'PASS'})
            extra['decomposition_tu_vs_saved_tu_max_abs']=float(np.max(abs(dec['entropy']-scores['entropy'])))
            for name in ['expected_entropy','mutual_information','gini_expected','gini_disagreement']:scores[name]=dec[name]
            extra['draw_count']=draws.shape[1]
    extra.update(label_hash=arrhash(y),ids_hash=arrhash(ids),n_samples=len(ids))
    return ids,p,y,z,raw_z,scores,extra,validation


def analyze_rank_unit(r,unit,loss,scores,groups,bootstrap=True,labels=None,label_names=None):
    records=[];curves=[];composition=[];repout={};cache={}
    weights,_=bootstrap_group_counts(groups,BOOT,PARAMS['analysis_seed']) if bootstrap else (None,None)
    oracle=RankPlan(loss,loss).evaluate()['aurc'][0]
    for name,u in scores.items():
        plan=RankPlan(u,loss);point=plan.evaluate()
        token=(arrhash(plan.order),arrhash(plan.starts))
        if bootstrap and token in cache:rep=cache[token]
        elif bootstrap:
            batches=[]
            for lo in range(0,BOOT,32):batches.append(plan.evaluate(weights[lo:lo+32]))
            rep={k:np.concatenate([b[k] for b in batches]) for k in batches[0]};cache[token]=rep
        else:rep={}
        curve=plan.evaluate(coverages=CURVE_Q)
        unique,counts=np.unique(u,return_counts=True)
        row={'object_id':r['object_id'],'dataset':r['dataset'],'model':r['model'],'adaptation':r['adaptation'],'method':r['method'],'seed':r['seed'],
             'analysis_unit':unit,'score':name,'n_units':len(loss),'n_groups':len(np.unique(groups)),
             'n_errors':int(np.sum(loss)) if plan.binary else np.nan,'binary_error_target':plan.binary,
             'auroc_ap_na_reason':'' if plan.binary and 0<np.sum(loss)<len(loss) else ('continuous_image_loss' if not plan.binary else 'constant_error_target'),
             'mean_uncertainty':float(np.mean(u)),'spearman_uncertainty_loss':correlation(u,loss),'oracle_aurc':oracle,
             'unique_scores':len(unique),'largest_tie':int(counts.max()),'tied_observation_fraction':float(counts[counts>1].sum()/len(u)),
             'evaluation_scope':'exploratory_fixed_models'}
        for metric,values in point.items():
            row[metric]=float(values[0])
            if bootstrap:
                lower,upper,nvalid=interval(rep[metric]);row[metric+'_ci_low']=lower;row[metric+'_ci_high']=upper;row[metric+'_valid_bootstraps']=nvalid
                repout[f'{unit}___{name}___{metric}']=rep[metric]
        records.append(row)
        for q in CURVE_Q:curves.append({'object_id':r['object_id'],'analysis_unit':unit,'score':name,'coverage':q,'risk':curve[f'risk_at_{q:g}'][0],'random_risk':point['base_risk'][0]})
        if labels is not None:
            for q in COVERAGES:
                accepted=plan.acceptance(q)
                if labels.ndim==1:
                    for c in range(len(label_names)):
                        mask=labels==c;support=int(mask.sum());ac=float(accepted[mask].sum())
                        composition.append({'object_id':r['object_id'],'analysis_unit':unit,'score':name,'coverage':q,'label':label_names[c],'positive_support':support,
                                            'accepted_mass':ac,'rejected_mass':support-ac,'within_class_acceptance':ac/support if support else np.nan})
                else:
                    for c in range(labels.shape[1]):
                        mask=labels[:,c]==1;support=int(mask.sum());ac=float(accepted[mask].sum())
                        composition.append({'object_id':r['object_id'],'analysis_unit':unit,'score':name,'coverage':q,'label':label_names[c],'positive_support':support,
                                            'accepted_mass':ac,'rejected_mass':support-ac,'within_class_acceptance':ac/support if support else np.nan})
    return records,curves,composition,repout


def core_class_metrics(p,y,z,binary):
    if binary:
        pred=p>=.5;err=pred!=y
        clipped=np.clip(p,np.finfo(np.float64).tiny,1-1e-15)
        prob_nll=float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean())
        metrics={'accuracy':float((~err).all(axis=1).mean()),'brier':float(((p-y)**2).mean()),'nll':prob_nll}
        if z is not None:metrics['nll_logits']=float((np.logaddexp(0,z)-y*z).mean())
        tp=np.sum(pred&(y==1),axis=0);fp=np.sum(pred&(y==0),axis=0);fn=np.sum(~pred&(y==1),axis=0)
        metrics['macro_f1']=float(np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(tp),dtype=float),where=2*tp+fp+fn>0).mean())
    else:
        pred=p.argmax(axis=1);err=pred!=y
        metrics={'accuracy':float((~err).mean()),'brier':float(((p-np.eye(p.shape[1])[y])**2).sum(axis=1).mean()),
                 'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],np.finfo(np.float64).tiny,1)).mean())}
        if z is not None:metrics['nll_logits']=float((logsumexp(z,axis=1)-z[np.arange(len(y)),y]).mean())
    return metrics,err


def compare_core(r,actual):
    expected,_=expected_row(r);rows=[]
    for name,value in actual.items():
        if name not in expected or expected[name]=='':continue
        old=float(expected[name]);delta=value-old
        rows.append({'metric':name,'actual':value,'expected':old,'difference':delta,'tolerance':2e-6,'status':'PASS' if abs(delta)<=2e-6 else 'INVESTIGATE'})
    return rows


def analyze_classification(r):
    dest=OUT/'objects'/r['object_id']
    if (dest/'done.json').exists():print('skip',r['object_id'],flush=True);return
    dest.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    ids,p,y,z,raw_z,scores,extra,validation=read_classification(r);loaded=time.perf_counter()
    groups=groups_for(r['dataset'],ids);binary=y.ndim==2
    if binary:
        names=yaml.safe_load((ROOT/'configs/treesatai_frozen_final.yaml').read_text())['data']['class_names']
    else:
        config=ROOT/f"configs/eurosat_{r['model']}_frozen_baseline.yaml"
        names=yaml.safe_load(config.read_text())['data']['class_names']
        with np.load(archive_path(r,'classification_ensemble_members'),allow_pickle=True) as d:
            if 'class_names' in d:assert names==d['class_names'].astype(str).tolist()
    actual,err=core_class_metrics(p,y,z,binary);checks=compare_core(r,actual)
    records=[];curves=[];composition=[];reps={};labelmetrics=[]
    if binary:
        image_loss=err.mean(axis=1);image_scores={k:v.mean(axis=1) for k,v in scores.items()}
        rr,cc,co,rp=analyze_rank_unit(r,'image_hamming',image_loss,image_scores,groups,True,y,names);records+=rr;curves+=cc;composition+=co;reps.update(rp)
        rr,cc,co,rp=analyze_rank_unit(r,'micro_label_decision',err.ravel(),{k:v.ravel() for k,v in scores.items()},np.repeat(groups,y.shape[1]),True);records+=rr;curves+=cc;reps.update(rp)
        for c,name in enumerate(names):
            for score,u in scores.items():
                plan=RankPlan(u[:,c],err[:,c]);m=plan.evaluate()
                labelmetrics.append({'object_id':r['object_id'],'label_index':c,'label':name,'score':score,'n_images':len(y),'positive_labels':int(y[:,c].sum()),'negative_labels':int(len(y)-y[:,c].sum()),
                                     'error_count':int(err[:,c].sum()),'correct_count':int((~err[:,c]).sum()),'auroc_ap_na_reason':'constant_error_target' if err[:,c].min()==err[:,c].max() else '',**{k:float(v[0]) for k,v in m.items()}})
        ld=pd.DataFrame(labelmetrics)
        macro=ld.groupby('score')[['auroc','average_precision','aurc','base_risk']].agg(['mean','count'])
        macro.columns=['_'.join(c) for c in macro.columns];macro.reset_index().to_csv(dest/'label_macro.csv',index=False)
        csvout(dest/'label_metrics.csv',labelmetrics)
    else:
        image_loss=err.astype(float);image_scores=scores
        rr,cc,co,rp=analyze_rank_unit(r,'image_top1',image_loss,scores,groups,True,y,names);records+=rr;curves+=cc;composition+=co;reps.update(rp)
    analysis_end=time.perf_counter()
    derived=dict(sample_id=ids,group_id=groups,label=y,probabilities=p,loss=image_loss,**{'score_'+k:v for k,v in scores.items()})
    if z is not None:derived['logits']=z
    if raw_z is not None:derived['raw_logits']=raw_z
    np.savez_compressed(dest/'derived.npz',**derived)
    np.savez_compressed(dest/'bootstrap.npz',**reps)
    csvout(dest/'metrics.csv',records);csvout(dest/'curves.csv',curves);csvout(dest/'class_coverage.csv',composition)
    scale=[];precision=[]
    if z is not None:
        if binary:
            for c,name in enumerate(names):scale.append({'label':name,'mean_log_odds':float(z[:,c].mean()),'mean_abs_log_odds':float(abs(z[:,c]).mean()),'spearman_abs_log_odds_error':correlation(abs(z[:,c]),err[:,c])})
        else:
            centered=z-z.mean(axis=1,keepdims=True);norm=np.linalg.norm(centered,axis=1);top=np.partition(z,-2,axis=1)[:,-2:];margin=top[:,1]-top[:,0]
            for name,values in [('centered_logit_norm',norm),('top1_top2_logit_margin',margin)]:
                scale.append({'measure':name,'mean':float(values.mean()),'mean_correct':float(values[~err].mean()),'mean_wrong':float(values[err].mean()),'spearman_error':correlation(values,err),'spearman_confidence':correlation(values,p.max(axis=1))})
            if r['method'] in ['deterministic','temperature_scaling']:
                reconstructed=softmax(z,axis=1);new_scores=base_scores(reconstructed)
                for name,u in new_scores.items():
                    old=RankPlan(scores[name],err).evaluate();new=RankPlan(u,err).evaluate()
                    precision.append({'score':name,'probability_max_abs_difference':float(np.max(abs(reconstructed-p))),'argmax_changes':int(np.sum(reconstructed.argmax(axis=1)!=p.argmax(axis=1))),
                                      **{k+'_saved':float(v[0]) for k,v in old.items()},**{k+'_logits64':float(v[0]) for k,v in new.items()}})
        if raw_z is not None:
            delta=float(np.max(abs(z-raw_z/extra['temperature'])));assert delta<2e-5
            validation.append({'check':'TS_scaled_logits_raw_over_T','max_abs_error':delta,'tolerance':2e-5,'status':'PASS'})
    csvout(dest/'scale.csv',scale);csvout(dest/'precision_sensitivity.csv',precision)
    decomposition_means=[{'score':name,'mean':float(v.mean()),'std_across_samples':float((v.mean(axis=1) if binary else v).std()),'score_semantics':'Bernoulli per-label then mean' if binary else 'categorical'} for name,v in scores.items()]
    proxy_correlations=[]
    for a in scores:
        for b in scores:
            if a>=b:continue
            aa=scores[a].mean(axis=1) if binary else scores[a];bb=scores[b].mean(axis=1) if binary else scores[b]
            proxy_correlations.append({'score_a':a,'score_b':b,'spearman':correlation(aa,bb),'interpretation':'proxy association, not source coupling'})
    csvout(dest/'proxy_means.csv',decomposition_means);csvout(dest/'proxy_correlations.csv',proxy_correlations)
    extra.update(actual_core_metrics=actual,core_checks=checks,validation=validation,
                 pmax_exactly_one=int(np.sum(p.max(axis=1)==1)) if not binary else int(np.sum((p==0)|(p==1))),
                 pmax_one_errors=int(np.sum(err[p.max(axis=1)==1])) if not binary else None,
                 load_seconds=loaded-start,ranking_bootstrap_seconds=analysis_end-loaded,total_seconds=time.perf_counter()-start,peak_rss_GiB=rss_gib(),status='COMPLETE')
    dump(dest/'done.json',extra)
    print('done',r['object_id'],'seconds',round(extra['total_seconds'],2),flush=True)


def boundary_mask(y,valid):
    b=np.zeros_like(valid)
    h=valid[:,1:]&valid[:,:-1]&(y[:,1:]!=y[:,:-1]);b[:,1:]|=h;b[:,:-1]|=h
    v=valid[1:,:]&valid[:-1,:]&(y[1:,:]!=y[:-1,:]);b[1:,:]|=v;b[:-1,:]|=v
    return maximum_filter(b,size=2*PARAMS['boundary_radius']+1,mode='constant')&valid


def analyze_segmentation(r):
    dest=OUT/'objects'/r['object_id']
    if (dest/'done.json').exists():print('skip',r['object_id'],flush=True);return
    dest.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    subset_manifest=json.loads((ROOT/'reports/mc_dropout_segmentation_research_subset_ids.json').read_text())
    subset_ids=subset_manifest['datasets'][r['dataset']]['final_test_research_subset_ids'];subset_set=set(subset_ids)
    names=['clear','thick_cloud','thin_cloud','cloud_shadow'] if r['dataset']=='cloudsen12' else ['background','building'];classes=len(names)
    confusion=np.zeros((classes,classes),dtype=np.int64);sums=dict(n=0,nll=0.,brier=0.)
    images=[];classrows=[];pixelrows=[];pixelstrata=[];pixelcurve=[];validation=[]
    raw=None;rawmap={};raw_mean_max=0.;gini_max=0.;saved_prediction_mismatches=0;full_var_mean_diff=0.;subset_label_matches=0
    with LazyNPZ(ROOT/r['prediction_path'],r['object_id']) as full:
        ids=full['sample_id'].astype(str);order=np.argsort(ids);ids=ids[order];groups=groups_for(r['dataset'],ids)
        # Numeric arrays are file-backed; only one image is converted to float64.
        pp=full['probabilities'];yy=full['label'];vv=full['valid_mask'];savedpred=full['prediction']
        mc_arrays={}
        if r['method']=='mc_dropout':
            for name,keyname in [('expected_entropy','expected_predictive_entropy'),('mutual_information','mi_style_disagreement'),('variance','predictive_variance')]:mc_arrays[name]=full[keyname]
        if r['raw_path']:
            raw=LazyNPZ(ROOT/r['raw_path'],r['object_id']+'_raw')
            rawids=raw['sample_id'].astype(str);rawmap={sid:i for i,sid in enumerate(rawids)}
            if list(rawids)!=subset_ids:raise ValueError('raw subset IDs differ from fixed definition')
            rawp=raw['probabilities'];rawy=raw['label'];rawv=raw['valid_mask'] if 'valid_mask' in raw else None
        loaded=time.perf_counter();sort_seconds=0.;aggregate_start=time.perf_counter()
        for j,i in enumerate(order):
            sid=ids[j];y=np.asarray(yy[i]);valid=np.asarray(vv[i],dtype=bool)
            if not np.array_equal(valid,y!=255):raise ValueError('saved valid mask differs from label-ignore contract')
            q=np.moveaxis(np.asarray(pp[i],dtype=np.float64),0,-1)[valid]
            lab=y[valid].astype(np.int64)
            if not len(lab):raise ValueError('image contains no valid pixels')
            if not np.isfinite(q).all() or q.min()<0 or q.max()>1 or np.max(abs(q.sum(axis=1)-1))>2e-6:raise ValueError('invalid segmentation probability')
            pred=q.argmax(axis=1);loss=(pred!=lab).astype(float);n=len(lab)
            saved_prediction_mismatches+=int(np.sum(pred!=np.asarray(savedpred[i])[valid]))
            scores=base_scores(q)
            if mc_arrays:
                scores['expected_entropy']=np.asarray(mc_arrays['expected_entropy'][i],dtype=np.float64)[valid]
                scores['mutual_information']=np.asarray(mc_arrays['mutual_information'][i],dtype=np.float64)[valid]
                scores['gini_disagreement']=np.moveaxis(np.asarray(mc_arrays['variance'][i],dtype=np.float64),0,-1)[valid].sum(axis=1)
                scores['gini_expected']=scores['gini_total']-scores['gini_disagreement']
            cm=np.bincount(classes*lab+pred,minlength=classes**2).reshape(classes,classes);confusion+=cm
            nll=float(-np.log(np.maximum(q[np.arange(n),lab],1e-12)).sum());brier=float(((q-np.eye(classes)[lab])**2).sum())
            sums['n']+=n;sums['nll']+=nll;sums['brier']+=brier
            images.append({'sample_id':sid,'group_id':groups[j],'valid_pixels':n,'pixel_error_rate':float(loss.mean()),'label_hash':arrhash(y),'valid_mask_hash':arrhash(valid),
                           'nll':nll/n,'brier':brier/n,**{'score_'+k:float(a.mean()) for k,a in scores.items()}})
            support=cm.sum(axis=1);intersection=cm.diagonal();union=cm.sum(axis=0)+support-intersection
            for c,name in enumerate(names):classrows.append({'sample_id':sid,'class_index':c,'class_name':name,'support':int(support[c]),'iou':intersection[c]/union[c] if union[c] else np.nan,
                                                            'recall':intersection[c]/support[c] if support[c] else np.nan,'union':int(union[c]),'na_reason':'absent_true_and_predicted_class' if union[c]==0 else ('no_true_class_for_recall' if support[c]==0 else '')})
            if sid not in subset_set:continue
            if raw is not None:
                k=rawmap[sid]
                if not np.array_equal(np.asarray(rawy[k]),y):raise ValueError('raw subset labels differ')
                rv=np.asarray(rawv[k],dtype=bool) if rawv is not None else (np.asarray(rawy[k])!=255)
                if not np.array_equal(rv,valid):raise ValueError('raw subset valid mask differs')
                subset_label_matches+=1
                pdraw=np.asarray(rawp[k]).transpose(2,3,0,1).reshape(-1,rawp.shape[1],classes)[valid.ravel()]
                mean,dec=decomposition(pdraw)
                diff=float(np.max(abs(mean-q)));raw_mean_max=max(raw_mean_max,diff)
                if diff>2e-6:raise ValueError(('subset mean mismatch',diff))
                gini_max=max(gini_max,float(np.max(abs(dec['gini_total']-dec['gini_expected']-dec['gini_disagreement']))))
                if mc_arrays:
                    full_var_mean_diff=max(full_var_mean_diff,float(np.max(abs(scores['gini_disagreement']-dec['gini_disagreement']))))
                    for name in ['expected_entropy','mutual_information']:
                        er=float(np.max(abs(scores[name]-dec[name])))
                        if er>2e-6:raise ValueError(('saved entropy mismatch',name,er))
                for name in ['expected_entropy','mutual_information','gini_expected','gini_disagreement']:scores[name]=dec[name]
                del pdraw,mean,dec
            bound=boundary_mask(y,valid)[valid]
            strata=[('boundary',bound),('nonboundary',~bound)]+[(name,lab==c) for c,name in enumerate(names)]
            t=time.perf_counter()
            for name,u in scores.items():
                plan=RankPlan(u,loss);m=plan.evaluate();curve=plan.evaluate(coverages=CURVE_Q)
                pixelrows.append({'sample_id':sid,'group_id':groups[j],'score':name,'valid_pixels':n,'errors':int(loss.sum()),'na_reason':'constant_error_target' if loss.min()==loss.max() else '',**{k:float(v[0]) for k,v in m.items()}})
                pixelcurve.append({'sample_id':sid,'score':name,**{f'{qv:.2f}':float(curve[f'risk_at_{qv:g}'][0]) for qv in CURVE_Q}})
                accepts={qv:plan.acceptance(qv) for qv in COVERAGES}
                for stratum,mask in strata:
                    count=int(mask.sum())
                    row={'sample_id':sid,'score':name,'stratum':stratum,'support':count,'error_rate':float(loss[mask].mean()) if count else np.nan,'mean_uncertainty':float(u[mask].mean()) if count else np.nan,'na_reason':'' if count else 'no_pixels_in_stratum'}
                    for qv,ac in accepts.items():row[f'accepted_mass_at_{qv:g}']=float(ac[mask].sum());row[f'within_stratum_coverage_at_{qv:g}']=float(ac[mask].mean()) if count else np.nan
                    pixelstrata.append(row)
            sort_seconds+=time.perf_counter()-t
        del pp,yy,vv,savedpred,mc_arrays
        if raw is not None:
            del rawp,rawy,rawv
            raw_seconds=raw.seconds;raw.close()
        else:raw_seconds=0.
        decompress_seconds=full.seconds+raw_seconds
        input_uncompressed=full.uncompressed_bytes
    aggregation_seconds=time.perf_counter()-aggregate_start-sort_seconds
    frame=pd.DataFrame(images);loss=frame.pixel_error_rate.to_numpy();image_scores={k[6:]:frame[k].to_numpy() for k in frame if k.startswith('score_')}
    bt=time.perf_counter();records,curves,composition,reps=analyze_rank_unit(r,'image_pixel_error_fraction',loss,image_scores,groups,True)
    bootstrap_seconds=time.perf_counter()-bt
    inter=confusion.diagonal();union=confusion.sum(axis=0)+confusion.sum(axis=1)-inter
    iou=np.divide(inter,union,out=np.full(classes,np.nan),where=union>0)
    actual={'pixel_accuracy':float(inter.sum()/sums['n']),'miou':float(np.nanmean(iou)),'nll':sums['nll']/sums['n'],'brier':sums['brier']/sums['n']}
    actual.update({f'iou_{name}':float(iou[c]) for c,name in enumerate(names)})
    checks=compare_core(r,actual)
    if raw is not None:
        assert subset_label_matches==32 and gini_max<2e-12 and full_var_mean_diff<2e-6
        validation=[{'check':'raw_subset_label_and_mask_alignment','matched_images':subset_label_matches,'status':'PASS'},
                    {'check':'raw_subset_mean_vs_full','max_abs_error':raw_mean_max,'tolerance':2e-6,'status':'PASS'},
                    {'check':'raw_gini_identity','max_abs_error':gini_max,'tolerance':2e-12,'status':'PASS'},
                    {'check':'full_MC_variance_vs_raw','max_abs_error':full_var_mean_diff,'tolerance':2e-6,'status':'PASS' if r['method']=='mc_dropout' else 'NA'}]
    np.savez_compressed(dest/'derived.npz',sample_id=ids,group_id=groups,loss=loss,**{'score_'+k:v for k,v in image_scores.items()})
    np.savez_compressed(dest/'bootstrap.npz',**reps)
    frame.to_csv(dest/'per_image.csv',index=False);csvout(dest/'per_image_class.csv',classrows)
    csvout(dest/'metrics.csv',records);csvout(dest/'curves.csv',curves);csvout(dest/'pixel_metrics.csv',pixelrows);csvout(dest/'pixel_strata.csv',pixelstrata)
    pc=pd.DataFrame(pixelcurve);pc.groupby('score').mean(numeric_only=True).to_csv(dest/'pixel_mean_curves.csv')
    ppix=pd.DataFrame(pixelrows);summaries=[];pixel_boot={}
    for score,g in ppix.groupby('score'):
        g=g.sort_values('sample_id');w,ug=bootstrap_group_counts(g.group_id.to_numpy(),BOOT,PARAMS['analysis_seed'])
        for metric in ['base_risk','aurc','auroc','average_precision']+[f'risk_at_{q:g}' for q in COVERAGES]:
            a=g[metric].to_numpy();valid=np.isfinite(a);numerator=w[:,valid]@a[valid];denominator=w[:,valid].sum(axis=1)
            draws=np.divide(numerator,denominator,out=np.full(BOOT,np.nan),where=denominator>0);lo,hi,nvalid=interval(draws)
            summaries.append({'object_id':r['object_id'],'score':score,'metric':metric,'mean':float(np.nanmean(a)) if valid.any() else np.nan,'total_images':len(g),'valid_images':int(valid.sum()),'groups':len(ug),'ci_low':lo,'ci_high':hi,'valid_bootstraps':nvalid,'scope':'fixed32_mean_of_per_image_pixel_metrics'})
            pixel_boot[f'{score}___{metric}']=draws
    csvout(dest/'pixel_summary.csv',summaries);np.savez_compressed(dest/'pixel_bootstrap.npz',**pixel_boot)
    proxy=[{'score':name,'mean':float(v.mean()),'std_across_samples':float(v.std()),'score_semantics':'mean over valid pixels, then images'} for name,v in image_scores.items()]
    csvout(dest/'proxy_means.csv',proxy)
    proxycor=[{'score_a':a,'score_b':b,'spearman':correlation(image_scores[a],image_scores[b]),'interpretation':'proxy association, not source coupling'} for a in image_scores for b in image_scores if a<b]
    csvout(dest/'proxy_correlations.csv',proxycor)
    done=dict(actual_core_metrics=actual,core_checks=checks,validation=validation,n_samples=len(ids),subset_images=32,ids_hash=arrhash(ids),label_hash=arrhash(frame.label_hash.to_numpy().astype(str)),mask_hash=arrhash(frame.valid_mask_hash.to_numpy().astype(str)),
              saved_prediction_mismatches=saved_prediction_mismatches,load_seconds=loaded-start,decompression_seconds=decompress_seconds,uncompressed_bytes_read=input_uncompressed,
              image_aggregation_seconds=aggregation_seconds,pixel_ranking_seconds=sort_seconds,ranking_bootstrap_seconds=bootstrap_seconds,total_seconds=time.perf_counter()-start,peak_rss_GiB=rss_gib(),status='COMPLETE')
    dump(dest/'done.json',done);print('done',r['object_id'],'seconds',round(done['total_seconds'],2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['classification','segmentation'],default='classification');parser.add_argument('--limit',type=int);args=parser.parse_args()
    rows=[r for r in get_inputs() if r['task']==args.task]
    if args.limit:rows=rows[:args.limit]
    for r in rows:
        if args.task=='classification':analyze_classification(r)
        else:analyze_segmentation(r)
        gc.collect()


if __name__=='__main__':main()
