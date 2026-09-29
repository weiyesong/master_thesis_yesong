"""Direct saved-array verification and prespecified diagnostic summaries."""
import json,time
import numpy as np
import pandas as pd
from common import ROOT,OUT,ARCHIVE,MASTER,get_inputs,csvout,dump,sha,arrhash
from metrics import decomposition,RankPlan,align_indices,COVERAGES


def main():
    rows=get_inputs();mc_checks=[];ts_checks=[];aux=[];coverage=[];core_de=[]
    for r in rows:
        dest=OUT/'objects'/r['object_id']
        if not (dest/'done.json').exists():raise RuntimeError(('missing object',r['object_id']))
        if r['raw_path']:
            p=ROOT/r['raw_path'];a=next(x for x in ARCHIVE if 'research_data/'+x['archive_path']==r['raw_path'])
            assert p.stat().st_size==a['size_bytes']
            aux.append({'object_id':r['object_id'],'path':r['raw_path'],'role':a['role'],'bytes':p.stat().st_size,'sha256_expected':a['sha256'],'hash_basis':'archive manifest; actual size and previously validated raw schema; NPZ CRC checked while reading'})
        if r['task']=='classification' and r['method']=='mc_dropout':
            binary=r['dataset']=='treesatai'
            with np.load(ROOT/r['raw_path'],allow_pickle=False) as raw,np.load(ROOT/r['prediction_path'],allow_pickle=False) as saved:
                rid=raw['sample_ids'].astype(str);sid=saved['sample_id'].astype(str);ix=align_indices(rid,sid);p=raw['probabilities'][ix]
                mean,d=decomposition(p,binary)
                comparisons={'mean_probabilities':mean,'predictive_variance':np.var(p.astype(np.float64),axis=1,ddof=0),
                             'predictive_entropy':d['entropy'].sum(axis=1) if binary else d['entropy'],
                             'expected_predictive_entropy':d['expected_entropy'].sum(axis=1) if binary else d['expected_entropy'],
                             'mi_style_disagreement':d['mutual_information'].sum(axis=1) if binary else d['mutual_information']}
                for name,values in comparisons.items():
                    delta=float(np.max(abs(values-saved[name])));assert delta<2e-6
                    mc_checks.append({'object_id':r['object_id'],'field':name,'max_abs_error':delta,'status':'PASS'})
        with np.load(dest/'derived.npz',allow_pickle=True) as z:
            data={name:z[name] for name in z.files}
        if r['method']=='temperature_scaling':
            drow=next(x for x in rows if x['dataset']==r['dataset'] and x['model']==r['model'] and x['adaptation']==r['adaptation'] and x['seed']==r['seed'] and x['method']=='deterministic')
            with np.load(OUT/'objects'/drow['object_id']/'derived.npz',allow_pickle=True) as d:
                assert np.array_equal(data['sample_id'],d['sample_id']) and np.array_equal(data['label'],d['label'])
                diff=float(np.max(abs(data['raw_logits']-d['logits'])));changes=int(np.sum(data['probabilities'].argmax(axis=1)!=d['probabilities'].argmax(axis=1)))
                assert diff<2e-6 and changes==0
                ts_checks.append({'object_id':r['object_id'],'baseline_object_id':drow['object_id'],'raw_logits_max_abs_error':diff,'argmax_changes':changes,'status':'PASS'})
        if r['task']=='segmentation':
            classes=pd.read_csv(dest/'per_image_class.csv')
            for sk,u in data.items():
                if not sk.startswith('score_'):continue
                plan=RankPlan(u,data['loss'])
                for q in COVERAGES:
                    accepted=plan.acceptance(q)
                    for name,g in classes.groupby('class_name'):
                        g=g.set_index('sample_id').loc[data['sample_id']];support=g.support.to_numpy();union=g.union.to_numpy()
                        tp=np.nan_to_num(g.iou.to_numpy())*union
                        s=float((support*accepted).sum());den=float((union*accepted).sum())
                        coverage.append({'object_id':r['object_id'],'dataset':r['dataset'],'model':r['model'],'adaptation':r['adaptation'],'method':r['method'],'score':sk[6:],'image_coverage':q,'class_name':name,
                                         'class_pixel_support':int(support.sum()),'retained_class_pixel_mass':s,'class_pixel_retention':s/support.sum() if support.sum() else np.nan,
                                         'retained_expected_confusion_iou':float((tp*accepted).sum()/den) if den else np.nan,
                                         'meaning':'fractional image acceptance; class diagnostics use ground truth only in evaluation'})
        if r['method']=='deep_ensemble':
            members=[x for x in MASTER if x['record_type']=='INDIVIDUAL_SEED' and x['uq_method']=='deterministic' and all(x[f]==r[f] for f in ['dataset','model','adaptation'])]
            ens=next(x for x in MASTER if x['record_type']=='ENSEMBLE' and all(x[f]==r[f] for f in ['dataset','model','adaptation']))
            assert len(members)==3
            for metric in ['accuracy','macro_f1','miou','pixel_accuracy','nll','brier','ece_15','iou_building']:
                if ens.get(metric,'')=='':continue
                vals=[float(m[metric]) for m in members]
                core_de.append({'object_id':r['object_id'],'dataset':r['dataset'],'model':r['model'],'adaptation':r['adaptation'],'metric':metric,'ensemble_value':float(ens[metric]),'mean_single_member_metric':np.mean(vals),'ensemble_minus_member_mean':float(ens[metric])-np.mean(vals),'member_seed_values':json.dumps(dict(zip([m['seed'] for m in members],vals))),'source':'reports/core_rq_completion_20260921/published/tables/thesis_master_results.csv','interpretation':'existing task/probability-quality metrics, not new member-mean RC'})
    csvout(OUT/'raw_classification_MC_verification.csv',mc_checks);csvout(OUT/'TS_same_checkpoint_verification.csv',ts_checks);csvout(OUT/'auxiliary_inputs_manifest.csv',aux)
    csvout(OUT/'full_image_class_coverage.csv',coverage);csvout(OUT/'DE_versus_three_member_core_metrics.csv',core_de)
    # Direct truth check against original EuroSAT split manifest.
    split=pd.read_csv(ROOT/'splits/eurosat_70_10_10_10_spatial20m/eurosat_splits.csv').set_index('sample_id');truth=[]
    for r in rows:
        if r['dataset']!='eurosat':continue
        with np.load(OUT/'objects'/r['object_id']/'derived.npz',allow_pickle=True) as z:
            match=np.array_equal(z['label'],split.loc[z['sample_id']].class_index.to_numpy());assert match
            truth.append({'object_id':r['object_id'],'labels_match_original_test_manifest':True,'all_split_test':bool((split.loc[z['sample_id']].split=='test').all())})
    csvout(OUT/'eurosat_original_label_verification.csv',truth)
    dump(OUT/'additional_verification_summary.json',{'MC_raw_summary_checks':len(mc_checks),'TS_same_checkpoint_pairs':len(ts_checks),'EuroSAT_truth_checks':len(truth),'all_passed':True,'auxiliary_paths':len(aux),'segmentation_member_mean_RC_scope':'Not executed beyond primary seed42. Three-member comparisons use existing core metrics; classification new RC has all three members.'})
    print('Additional direct verifications complete')


if __name__=='__main__':main()
