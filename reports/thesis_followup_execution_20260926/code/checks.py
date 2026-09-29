import json,csv,hashlib
import pandas as pd
from common import ROOT,OUT,get_inputs,csvout,dump,sha


def main():
    rows=get_inputs();application=[]
    for r in rows:
        scopes=['classification_full'] if r['task']=='classification' else ['segmentation_full_image','segmentation_fixed32_pixel']
        for scope in scopes:
            for score in ['msp','entropy','negative_margin','gini_total','expected_entropy','mutual_information','gini_expected','gini_disagreement']:
                component=score in ['expected_entropy','mutual_information','gini_expected','gini_disagreement']
                available=not component or r['method']=='mc_dropout' or (r['method']=='deep_ensemble' and scope!='segmentation_full_image')
                reason='' if available else ('no empirical prediction-distribution decomposition for a single deterministic prediction' if r['method']!='deep_ensemble' else 'full-test DE decomposition intentionally disabled; available on common fixed32 subset')
                application.append({'object_id':r['object_id'],'scope':scope,'quantity':score,'status':'PASS' if available else 'NA','reason':reason,'evidence_path':f"objects/{r['object_id']}/metrics.csv" if scope!='segmentation_fixed32_pixel' else f"objects/{r['object_id']}/pixel_metrics.csv"})
            if scope=='segmentation_full_image' or r['dataset']=='treesatai':
                application.append({'object_id':r['object_id'],'scope':scope,'quantity':'image_level_AUROC_AP','status':'NA','reason':'image loss is continuous Hamming/pixel-error fraction; binary error AUROC not its estimand','evidence_path':f"objects/{r['object_id']}/metrics.csv"})
    for evidence, part in pd.DataFrame(application).query("status == 'PASS'").groupby('evidence_path'):
        present=set(pd.read_csv(OUT/evidence).score)
        assert set(part.quantity).issubset(present),(evidence,set(part.quantity)-present)
    csvout(OUT/'analysis_applicability.csv',application)
    with (ROOT/'reports/core_rq_completion_20260921/checks.csv').open() as f:
        old=list(csv.DictReader(f))
    additional={
        'C01':'ANALYSIS_PROTOCOL.md;source_snapshot.json',
        'C02':'TS_same_checkpoint_verification.csv;inputs_manifest.csv',
        'C03':'scientific_tests.log;label_metrics.csv;analysis_applicability.csv',
        'C05':'core_metric_verification.csv;scientific_tests.log;precision_sensitivity.csv',
        'C06':'alignment_verification.csv;auxiliary_inputs_manifest.csv;eurosat_original_label_verification.csv',
        'C07':'ANALYSIS_PROTOCOL.md;seed_descriptive_summary.csv;paired_effects.csv',
        'C08':'sample_groups.csv;grouping_summary.json',
        'C09':'analysis_applicability.csv;ANALYSIS_PROTOCOL.md',
        'R1.4':'full_image_class_coverage.csv;label_metrics.csv',
        'R3.1':'inputs_manifest.csv;alignment_verification.csv;paired_effects.csv',
        'R3.3':'TS_same_checkpoint_verification.csv;TS_calibration_behavior.csv',
        'R3.4':'raw_classification_MC_verification.csv;core_metric_verification.csv',
        'R3.5':'DE_versus_three_member_core_metrics.csv;core_metric_verification.csv',
        'R3.7':'seed_descriptive_summary.csv;ANALYSIS_PROTOCOL.md'}
    output=[]
    for r in old:
        cid=r.get('check_id',r.get('id'));evidence=r.get('evidence_path','')
        output.append({'check_id':cid,'rq':r.get('rq',''),'status':r['status'],
                       'evidence_path':additional.get(cid,'../core_rq_completion_20260921/checks.csv'),
                       'legacy_direct_evidence_reference':'../core_rq_completion_20260921/checks.csv#'+str(cid),
                       'legacy_evidence_paths':evidence,
                       'review_basis':'new direct analysis plus preserved prior evidence' if cid in additional else 'unchanged prior direct evidence; not rerun this phase',
                       'finding':r.get('finding',''),'impact_on_answer':'DOFA–EuroSAT historical provenance remains bounded UNKNOWN' if cid in ['C01','C06'] else 'Original core RQ scope unchanged; see new A/E findings separately',
                       'minimal_resolution':r['minimal_resolution'],
                       'legacy_impact_on_answer':r['impact_on_answer'],
                       'legacy_acceptance_evidence':r['acceptance_evidence'],
                       'owner':'Codex; Claude independent sample review'})
    extras=[
        ('FUP01','A coverage','PASS','inputs_manifest.csv;aggregation_summary.json','68 classification + 32 primary segmentation objects completed','none'),
        ('FUP02','A measurement','PASS','scientific_tests.log;core_metric_verification.csv;claude_code_review.md','Mathematical tests, direct primary-metric reconstruction, and independent sample checks','none'),
        ('FUP03','A pairing','PASS','alignment_verification.csv;TS_same_checkpoint_verification.csv;raw_classification_MC_verification.csv','Direct IDs/labels/masks/means and same-checkpoint TS verified','none'),
        ('FUP04','A inference scope','PASS','grouping_summary.json;paired_effects.csv;pixel_paired_effects.csv','Conditional fixed-model group bootstrap; pixel pairs use common-valid images','none'),
        ('FUP05','A source and definition precision','PASS','precision_sensitivity.csv;analysis_applicability.csv;ANALYSIS_PROTOCOL.md','Saved float32 tie structure retained; logits64 sensitivity separately labeled; task-specific NA','none'),
        ('FUP06','E analytic validity','PASS','toy/verification.json;toy/condition_means.csv;toy/interactions.csv','888 weighted terms, four conditions; identities and quadrature agree','none'),
        ('FUP07','A/E claim boundaries','PASS','FOLLOWUP_COMPLETION_REPORT.md;THESIS_RESULTS_DRAFT.md','No error-ranking champion, physical source-coupling or novel-theory claim inferred','none'),
        ('FUP08','unselected packages','NA','parameters.json;ANALYSIS_PROTOCOL.md','B/C/D/F, extra segmentation seeds, full-test DE decomposition, inference benchmark disabled','Enable a separate branch only if a new claim requires it')]
    for cid,rq,status,ev,finding,remedy in extras:output.append({'check_id':cid,'rq':rq,'status':status,'evidence_path':ev,'legacy_direct_evidence_reference':'','legacy_evidence_paths':'','review_basis':'this execution direct evidence','finding':finding,'impact_on_answer':rq,'minimal_resolution':remedy,'owner':'Codex; Claude independent sample review'})
    for item in output:
        item['evidence_path']=item['evidence_path'].replace('scientific_tests.log','scientific_tests_final.log')
    csvout(OUT/'checks_delta.csv',output)
    snapshot=json.loads((OUT/'source_snapshot.json').read_text());unchanged=[]
    for path,digest in snapshot['sha256'].items():assert sha(ROOT/path)==digest,path;unchanged.append(path)
    for r in rows:
        p=ROOT/r['prediction_path'];assert p.stat().st_size==int(r['bytes']) and p.stat().st_mtime_ns==int(r['mtime_ns']),str(p)
    assert sha(OUT/'ANALYSIS_PROTOCOL.md')==snapshot['protocol_sha256'],'Frozen protocol changed'
    group_sources=[]
    for dataset,entry in json.loads((OUT/'grouping_summary.json').read_text()).items():
        path='splits/eurosat_70_10_10_10_spatial20m/eurosat_splits.csv' if dataset=='eurosat' else f'reports/dataset_manifests/{dataset}_actual_manifest.csv'
        assert sha(ROOT/path)==entry['source_sha256'];group_sources.append(path)
    dump(OUT/'input_integrity.json',{'snapshot_sources_unchanged':unchanged,'prediction_files_size_and_mtime_unchanged':len(rows),'original_source_hash_checks':len(unchanged),'grouping_sources_hash_unchanged':group_sources,'frozen_protocol_hash_unchanged':True,'full_archive_rehashed':False})
    print('Wrote',len(output),'audit delta rows')


if __name__=='__main__':main()
