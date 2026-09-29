"""Verify the completed delivery and seal file hashes after independent review."""
import json,re,py_compile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,OUT,PARAMS,get_inputs,csvout,dump,sha


def main():
    rows=get_inputs();assert len(rows)==100
    assert Counter(r['task'] for r in rows)=={'classification':68,'segmentation':32}
    assert set(PARAMS['enabled_packages'])=={'A','E'}
    for flag in ['A_segmentation_extra_seeds','A_segmentation_full_test_DE_decomposition','E_sampling_illustration','timing_benchmark']:
        assert PARAMS[flag] is False,flag
    for r in rows:assert json.loads((OUT/'objects'/r['object_id']/'done.json').read_text())['status']=='COMPLETE'
    core=pd.read_csv(OUT/'core_metric_verification.csv',keep_default_na=False)
    assert Counter(core.status)=={'PASS':544,'NA':8}
    scalar=core[core.metric!=''];assert len(scalar)==436
    maxerr=float(pd.to_numeric(scalar.difference).abs().max())
    assert maxerr<2e-6
    alignment=pd.read_csv(OUT/'alignment_verification.csv');assert len(alignment)==100 and (alignment.status=='PASS').all()
    mc=pd.read_csv(OUT/'raw_classification_MC_verification.csv');assert len(mc)==60 and (mc.status=='PASS').all()
    ts=pd.read_csv(OUT/'TS_same_checkpoint_verification.csv');assert len(ts)==12 and (ts.status=='PASS').all() and ts.argmax_changes.sum()==0
    truth=pd.read_csv(OUT/'eurosat_original_label_verification.csv');assert len(truth)==40 and truth.labels_match_original_test_manifest.all() and truth.all_split_test.all()
    assert len(pd.read_csv(OUT/'inputs_lineage.csv'))==100
    legacy=pd.read_csv(ROOT/'reports/core_rq_completion_20260921/checks.csv',keep_default_na=False)
    delta=pd.read_csv(OUT/'checks_delta.csv',keep_default_na=False)
    assert dict(zip(legacy.check_id,legacy.status))==dict(zip(delta.check_id[:29],delta.status[:29]))
    assert Counter(delta.status[:29])=={'PASS':27,'UNKNOWN':2}
    assert Counter(delta.status[29:])=={'PASS':7,'NA':1}
    tests=(OUT/'scientific_tests_final.log').read_text();assert 'Ran 10 tests' in tests and tests.rstrip().endswith('OK')
    for p in (OUT/'code').glob('*.py'):py_compile.compile(str(p),doraise=True)
    figs=json.loads((OUT/'figures/figure_manifest.json').read_text());assert len(figs)==11
    for f in figs:
        for suffix in ['pdf','png']:assert (OUT/'figures'/f"{f['figure']}.{suffix}").stat().st_size>1000
        for source in f['sources']:assert (OUT/source).is_file()
    review=json.loads((OUT/'claude_final_raw.json').read_text());assert not review['is_error'] and review['subtype']=='success'
    assert review['session_id']=='5afb33d9-4dac-4ed1-8d17-2e1ec2a65b13'
    assert (OUT/'claude_final_review.md').read_text().strip()==review['result'].strip()
    trace=[json.loads(l) for l in (OUT/'claude_evidence_trace.jsonl').read_text().splitlines()]
    assert all(r['block']['type'] in ['tool_use','tool_result'] for r in trace)
    links=[]
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            clean=target.split('#',1)[0];clean=re.sub(r':\d+$','',clean).strip('<>')
            q=p.parent/clean
            links.append({'document':p.name,'target':target,'exists':q.exists()})
    assert all(x['exists'] for x in links),[x for x in links if not x['exists']]
    csvout(OUT/'artifact_link_verification.csv',links)
    m=pd.read_csv(OUT/'metrics.csv');wi=pd.read_csv(OUT/'within_predictor_score_effects.csv')
    retention=pd.read_csv(OUT/'full_image_class_coverage.csv')
    ret=retention[(retention.dataset=='spacenet7')&(retention.score=='msp')&(retention.image_coverage==.5)&(retention.class_name=='building')]
    assert len(ret)==16
    mi=wi[(wi.dataset=='treesatai')&(wi.analysis_unit=='micro_label_decision')&(wi.score=='mutual_information')&(wi.metric=='auroc')]
    assert len(mi)==10 and (mi.point_difference<0).all() and (mi.ci_high<0).all()
    tc=pd.read_csv(OUT/'TS_calibration_behavior.csv');assert len(tc)==12 and (tc.delta_ece<0).sum()==8
    facts={'created_utc':datetime.now(timezone.utc).isoformat(),'objects':len(rows),'coverage_by_dataset_method':{str(k):int(v) for k,v in pd.DataFrame(rows).groupby(['dataset','method']).size().items()},
           'scalar_checks':len(scalar),'maximum_primary_metric_difference':maxerr,'core_verification_status':dict(Counter(core.status)),
           'alignment_objects':len(alignment),'MC_raw_classification_checks':len(mc),'TS_same_checkpoint_pairs':len(ts),'EuroSAT_original_truth_objects':len(truth),'scientific_tests':10,'standalone_figures':len(figs),
           'TreeSat_MI_minus_MSP_micro_AUROC_range':[float(mi.point_difference.min()),float(mi.point_difference.max())],
           'SpaceNet7_building_retention_at_half_image_coverage_range':[float(ret.class_pixel_retention.min()),float(ret.class_pixel_retention.max())],
           'TS_ECE_improved_pairs':int((tc.delta_ece<0).sum()),'markdown_links_checked':len(links),'all_links_exist':True,
           'old_checks':dict(Counter(delta.status[:29])),'new_checks':dict(Counter(delta.status[29:])),'model_training_runs':0,'model_forward_runs':0}
    dump(OUT/'final_verification.json',facts)
    required=['ANALYSIS_PROTOCOL.md','parameters.json','inputs_manifest.csv','inputs_lineage.csv','analysis_applicability.csv','source_snapshot.json','input_integrity.json','code_revision.json',
              'analysis_code.patch','implementation_changes.patch','PROTOCOL_CHANGELOG.md','DATA_DICTIONARY.md','FOLLOWUP_COMPLETION_REPORT.md','THESIS_RESULTS_DRAFT.md','REVIEW_RESOLUTION.md',
              'checks_delta.csv','resource_profile.csv','resource_summary.json','execution_environment.json','claude_session_metadata.json','claude_evidence_trace.jsonl','final_verification.json','README.md']
    assert all((OUT/p).is_file() for p in required)
    assert not list((OUT/'scratch').iterdir()),'Analysis scratch not empty'
    own_review_scratch=OUT/'claude_validation/scratch'
    assert not own_review_scratch.exists() or not list(own_review_scratch.iterdir()),'Review scratch not empty'
    outputs=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name=='completion_manifest.json':continue
        outputs.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    dump(OUT/'completion_manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETE_A_E_WITH_DECLARED_HISTORICAL_LIMITATIONS',
         'required_deliverables':required,'file_count':len(outputs),'total_bytes':sum(x['bytes'] for x in outputs),'files':outputs,'manifest_self_hash_excluded':True,
         'independent_review_scope':'actual Claude CLI independent representative raw-data checks; not a second full run of all objects'})
    print(json.dumps(facts,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
