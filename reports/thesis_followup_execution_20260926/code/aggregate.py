"""Combine completed A/E results and construct paired, fixed-model contrasts."""
from __future__ import annotations
import argparse,json
import numpy as np
import pandas as pd
from common import OUT,get_inputs,dump,csvout
from metrics import interval,bootstrap_group_counts

METRICS=['base_risk','aurc','auroc','average_precision','risk_at_1','risk_at_0.9','risk_at_0.8','risk_at_0.5']


def k(r,method=None,seed=None,adaptation=None,model=None):
    return (r['dataset'],model or r['model'],adaptation or r['adaptation'],method or r['method'],str(r['seed'] if seed is None else seed))


def readcsv(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    rows=get_inputs();done=[r for r in rows if (OUT/'objects'/r['object_id']/'done.json').exists()]
    if len(done)!=len(rows) and not args.allow_partial:raise RuntimeError(f'Only {len(done)}/{len(rows)} complete')
    lookup={k(r):r for r in done};metadata={r['object_id']:r for r in done};point={};replicates={};derived={}
    tables={name:[] for name in ['metrics','curves','class_coverage','label_metrics','label_macro','scale','precision_sensitivity','proxy_means','proxy_correlations','pixel_summary']}
    checks=[];profiles=[];alignment=[]
    for r in done:
        dest=OUT/'objects'/r['object_id'];summary=json.loads((dest/'done.json').read_text())
        for name in tables:
            path=dest/(name+'.csv')
            if not path.exists():continue
            frame=readcsv(path)
            if frame.empty:continue
            for field in ['object_id','task','dataset','model','adaptation','method','seed']:frame[field]=r[field]
            frame['scope']='exploratory_fixed_predictions';tables[name].append(frame)
        frame=readcsv(dest/'metrics.csv')
        point[r['object_id']]={(v['analysis_unit'],v['score']):v for v in frame.to_dict('records')}
        with np.load(dest/'bootstrap.npz',allow_pickle=False) as z:replicates[r['object_id']]={a:z[a] for a in z.files}
        # Early execution-owned tables serialized group IDs as object strings.
        with np.load(dest/'derived.npz',allow_pickle=True) as z:derived[r['object_id']]={a:z[a] for a in z.files}
        for check in summary.get('core_checks',[])+summary.get('validation',[]):checks.append({'object_id':r['object_id'],'dataset':r['dataset'],**check})
        profiles.append({**{f:r[f] for f in ['object_id','task','dataset','method']},**{key:value for key,value in summary.items() if key.endswith('_seconds') or key in ['peak_rss_GiB','uncompressed_bytes_read','n_samples']}})
    for name,frames in tables.items():
        if frames:pd.concat(frames,ignore_index=True).to_csv(OUT/(name+'.csv'),index=False)
    # Verify every object's actual derived ID/label identity against same-task reference.
    for ds in sorted({r['dataset'] for r in done}):
        rs=[r for r in done if r['dataset']==ds];reference=rs[0];ref=derived[reference['object_id']]
        refdone=json.loads((OUT/'objects'/reference['object_id']/'done.json').read_text())
        for r in rs:
            d=derived[r['object_id']];sd=json.loads((OUT/'objects'/r['object_id']/'done.json').read_text())
            idok=np.array_equal(d['sample_id'],ref['sample_id']);groupok=np.array_equal(d['group_id'],ref['group_id'])
            labelok=np.array_equal(d['label'],ref['label']) if 'label' in d else sd['label_hash']==refdone['label_hash']
            maskok=True if 'label' in d else sd['mask_hash']==refdone['mask_hash']
            assert idok and groupok and labelok and maskok,(r['object_id'],'alignment')
            alignment.append({'object_id':r['object_id'],'reference_object':reference['object_id'],'ids_equal':idok,'groups_equal':groupok,'labels_equal':labelok,'masks_equal':maskok,'status':'PASS'})
    contrasts=[]
    def add(r,targets,name):
        if all(t in lookup for t in targets):contrasts.append((r,[lookup[t] for t in targets],name))
    for r in done:
        if r['method']=='temperature_scaling':add(r,[k(r,method='deterministic')],'TS_minus_D')
        if r['method']=='mc_dropout':
            add(r,[k(r,method='mc_dropout_off')],'MC_minus_Off');add(r,[k(r,method='deterministic')],'MC_minus_D')
        if r['method']=='mc_dropout_off':add(r,[k(r,method='deterministic')],'Off_minus_D')
        if r['method']=='deep_ensemble':
            add(r,[k(r,method='deterministic',seed='42')],'DE_minus_D42')
            add(r,[k(r,method='deterministic',seed=str(s)) for s in [42,43,44]],'DE_minus_mean_D_members')
        if r['adaptation']=='full_finetune':add(r,[k(r,adaptation='frozen')],'full_recipe_minus_frozen_recipe')
        if r['model']=='panopticon':add(r,[k(r,model='dofa')],'Pan_configuration_minus_DOFA_configuration')
    paired=[];proxy=[]
    for r,baselines,contrast in contrasts:
        oid=r['object_id'];bids=[b['object_id'] for b in baselines]
        for (unit,score),row in point[oid].items():
            if not all((unit,score) in point[b] for b in bids):continue
            for metric in METRICS:
                bvalue=float(np.mean([point[b][(unit,score)][metric] for b in bids]));value=row[metric]-bvalue
                rk=f'{unit}___{score}___{metric}';dist=replicates[oid][rk]-np.mean([replicates[b][rk] for b in bids],axis=0)
                lo,hi,n=interval(dist)
                paired.append({'object_id':oid,'baseline_object_ids':';'.join(bids),'dataset':r['dataset'],'model':r['model'],'adaptation':r['adaptation'],'method':r['method'],'seed':r['seed'],
                               'contrast':contrast,'analysis_unit':unit,'score':score,'metric':metric,'point_difference':value,'ci_low':lo,'ci_high':hi,'valid_bootstraps':n,'estimand':'fixed predictors; evaluation-group bootstrap'})
        d=derived[oid];w,_=bootstrap_group_counts(d['group_id'])
        for name in d:
            if not name.startswith('score_') or not all(name in derived[b] for b in bids):continue
            a=d[name];a=a.mean(axis=1) if a.ndim==2 else a
            b=[]
            for bid in bids:
                v=derived[bid][name];b.append(v.mean(axis=1) if v.ndim==2 else v)
            delta=a-np.mean(b,axis=0);dist=(w@delta)/w.sum(axis=1);lo,hi,n=interval(dist)
            proxy.append({'object_id':oid,'baseline_object_ids':';'.join(bids),'dataset':r['dataset'],'model':r['model'],'adaptation':r['adaptation'],'method':r['method'],'seed':r['seed'],'contrast':contrast,'score':name[6:],'mean_difference':delta.mean(),'ci_low':lo,'ci_high':hi,'valid_bootstraps':n,'interpretation':'model/estimator-dependent proxy; no source identification'})
    within=[]
    for r in done:
        oid=r['object_id']
        for (unit,score),row in point[oid].items():
            if score=='msp':continue
            base=point[oid][(unit,'msp')]
            for metric in METRICS:
                dist=replicates[oid][f'{unit}___{score}___{metric}']-replicates[oid][f'{unit}___msp___{metric}']
                lo,hi,n=interval(dist)
                within.append({'object_id':oid,'dataset':r['dataset'],'model':r['model'],'adaptation':r['adaptation'],'method':r['method'],'seed':r['seed'],'analysis_unit':unit,'score':score,'baseline_score':'msp','metric':metric,'point_difference':row[metric]-base[metric],'ci_low':lo,'ci_high':hi,'valid_bootstraps':n,'estimand':'same predictor, paired evaluation groups'})
    # Pixel comparisons use intersection of finite per-image metrics for each pair.
    pixelpairs=[]
    for r,baselines,contrast in contrasts:
        if r['task']!='segmentation' or len(baselines)!=1:continue
        a=readcsv(OUT/'objects'/r['object_id']/'pixel_metrics.csv');b=readcsv(OUT/'objects'/baselines[0]['object_id']/'pixel_metrics.csv')
        merged=a.merge(b,on=['sample_id','score'],suffixes=('_a','_b'),validate='one_to_one')
        for score,g in merged.groupby('score'):
            for metric in METRICS:
                valid=np.isfinite(g[metric+'_a'])&np.isfinite(g[metric+'_b']);v=g[valid].sort_values('sample_id')
                if not len(v):continue
                w,ug=bootstrap_group_counts(v.group_id_a.to_numpy());delta=v[metric+'_a'].to_numpy()-v[metric+'_b'].to_numpy();dist=(w@delta)/w.sum(axis=1);lo,hi,n=interval(dist)
                pixelpairs.append({'object_id':r['object_id'],'baseline_object_id':baselines[0]['object_id'],'dataset':r['dataset'],'contrast':contrast,'score':score,'metric':metric,'mean_paired_difference':float(delta.mean()),'ci_low':lo,'ci_high':hi,'valid_bootstraps':n,'common_valid_images':len(v),'total_subset_images':len(g),'groups':len(ug),'estimand':'common-valid fixed32 images; paired group-bootstrap mean of per-image pixel metrics'})
    csvout(OUT/'core_metric_verification.csv',checks);csvout(OUT/'resource_profile.csv',profiles);csvout(OUT/'alignment_verification.csv',alignment)
    csvout(OUT/'paired_effects.csv',paired);csvout(OUT/'within_predictor_score_effects.csv',within);csvout(OUT/'proxy_paired_effects.csv',proxy);csvout(OUT/'pixel_paired_effects.csv',pixelpairs)
    # Seed summaries are descriptive. Ensembles are not multiplied into 3 replicates.
    metrics=pd.concat(tables['metrics'],ignore_index=True)
    seedsummary=metrics.groupby(['dataset','model','adaptation','method','analysis_unit','score'])[METRICS].agg(['mean','std','min','max','count'])
    seedsummary.columns=['_'.join(c) for c in seedsummary.columns];seedsummary.reset_index().to_csv(OUT/'seed_descriptive_summary.csv',index=False)
    dump(OUT/'aggregation_summary.json',{'complete_objects':len(done),'planned_objects':len(rows),'paired_effect_rows':len(paired),'score_effect_rows':len(within),'pixel_effect_rows':len(pixelpairs),'all_completed_objects_id_label_mask_alignment':True})
    print('Aggregated',len(done),'objects;',len(paired),'paired effects')


if __name__=='__main__':main()
