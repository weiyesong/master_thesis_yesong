"""Package metadata only: no neural evaluation or new scientific estimand."""
import csv,json,os,platform,subprocess,sys,zipfile,difflib
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import yaml
from common import ROOT,OUT,MASTER,get_inputs,sha,csvout,dump


def main():
    inputs=get_inputs();lineage=[];raw_bytes=[]
    groups=pd.read_csv(OUT/'sample_groups.csv')
    for r in inputs:
        p=ROOT/r['prediction_path']
        if p.suffix=='.parquet':fields=pq.read_schema(p).names
        else:
            with zipfile.ZipFile(p) as z:fields=[n[:-4] for n in z.namelist() if n.endswith('.npy')]
        label=next(n for n in ['label','true_label','true_labels'] if n in fields)
        idfield=next(n for n in ['sample_id','sample_ids'] if n in fields)
        prob=next(n for n in ['mean_probabilities','probabilities'] if n in fields)
        if r['dataset']=='eurosat':config=f"configs/eurosat_{r['model']}_frozen_baseline.yaml"
        else:config=f"configs/{r['dataset']}_frozen_final.yaml"
        names=yaml.safe_load((ROOT/config).read_text())['data']['class_names']
        if r['method']=='deep_ensemble':
            refs=[m for m in MASTER if m['record_type']=='INDIVIDUAL_SEED' and m['uq_method']=='deterministic' and all(m[f]==r[f] for f in ['dataset','model','adaptation'])]
            assert len(refs)==3
        else:
            meth='mc_dropout' if r['method']=='mc_dropout_off' else r['method']
            refs=[m for m in MASTER if m['record_type']=='INDIVIDUAL_SEED' and m['uq_method']==meth and all(m[f]==r[f] for f in ['dataset','model','adaptation','seed'])]
            assert len(refs)==1
            assert refs[0]['checkpoint_path']==r['checkpoint_path']
        paths=[x['checkpoint_path'] for x in refs]
        assert all((ROOT/x).is_file() for x in paths)
        lineage.append({'object_id':r['object_id'],'prediction_path':r['prediction_path'],'label_path':r['prediction_path'],'id_field':idfield,'label_field':label,'probability_field':prob,'valid_mask_field':'valid_mask' if 'valid_mask' in fields else 'NA_classification',
                        'class_names_json':json.dumps(names),'class_order_source':config,'class_order_source_sha256':sha(ROOT/config),
                        'groups_path':str((OUT/'sample_groups.csv').relative_to(ROOT)),'group_metadata_source':groups.loc[groups.dataset==r['dataset'],'source_path'].iloc[0],
                        'run_ids_json':json.dumps([x['run_id'] for x in refs]),'checkpoint_paths_json':json.dumps(paths),'member_seeds_json':json.dumps([x['seed'] for x in refs]),
                        'lineage_source':'reports/core_rq_completion_20260921/published/tables/thesis_master_results.csv','checkpoint_lineage_status':'PASS_paths_and_existing_registry; historical provenance limitations unchanged'})
        if r['task']=='segmentation' and r['raw_path']:
            with zipfile.ZipFile(ROOT/r['raw_path']) as z:
                selected=[n for n in ['sample_id.npy','probabilities.npy','label.npy','valid_mask.npy'] if n in z.namelist()]
                raw_bytes.append({'object_id':r['object_id'],'arrays':selected,'uncompressed_npy_bytes':sum(z.getinfo(n).file_size for n in selected),'basis':'NPZ member headers for arrays actually read; includes NPY headers'})
    csvout(OUT/'inputs_lineage.csv',lineage)
    rp=pd.read_csv(OUT/'resource_profile.csv')
    dump(OUT/'resource_summary.json',{'sum_object_wall_seconds':float(rp.total_seconds.sum()),'by_task_wall_seconds':rp.groupby('task').total_seconds.sum().to_dict(),
         'peak_process_rss_GiB':float(rp.peak_rss_GiB.max()),'segmentation_primary_uncompressed_npy_bytes':int(rp.uncompressed_bytes_read.sum()),
         'segmentation_raw_uncompressed_npy_bytes':sum(x['uncompressed_npy_bytes'] for x in raw_bytes),'raw_byte_details':raw_bytes,
         'prediction_compressed_file_bytes':sum(int(r['bytes']) for r in inputs),'cost_scope':'sum of per-object processing wall times; excludes implementation, independent review, aggregation, plotting, and packaging; not job elapsed or GPU time',
         'concurrency':'initial single-object profiles sequential; classification and segmentation subsequently overlapped, each task single process with BLAS/OMP threads=1',
         'toy_cpu_wall_seconds':json.loads((OUT/'toy/verification.json').read_text())['seconds']})
    env={'captured_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),
         'packages':{n:version(n) for n in ['numpy','scipy','pandas','pyarrow','matplotlib','scikit-learn','PyYAML']},
         'thread_env_expected':{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},'model_training_runs':0,'model_forward_runs':0}
    env['cpu_model']=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),None)
    env['physical_memory_kB']=int(next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal:')).split()[1])
    dump(OUT/'execution_environment.json',env)
    git=subprocess.run(['git','rev-parse','--verify','HEAD'],cwd=ROOT,capture_output=True,text=True)
    status=subprocess.run(['git','status','--porcelain=v1','--untracked-files=normal'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'workspace_git_status.txt').write_text(status.stdout+status.stderr)
    files=sorted((OUT/'code').glob('*.py'));revision={str(p.relative_to(OUT)):sha(p) for p in files}
    dump(OUT/'code_revision.json',{'git_head':git.stdout.strip() if git.returncode==0 else None,'git_head_status':'available' if git.returncode==0 else 'unborn repository; no tracked files; no historical diff is available',
         'files_sha256':revision,'revision_basis':'content hashes of complete delivered analysis source','frozen_protocol_sha256':sha(OUT/'ANALYSIS_PROTOCOL.md')})
    patches=[];changes=[]
    for p in files:
        rel=str(p.relative_to(OUT));patches.extend(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=rel))
        before=OUT/'code_versions/pre_bool_correlation_fix'/p.name
        if before.is_file():changes.extend(difflib.unified_diff(before.read_text().splitlines(True),p.read_text().splitlines(True),fromfile=str(before.relative_to(OUT)),tofile=rel))
    (OUT/'analysis_code.patch').write_text(''.join(patches));(OUT/'implementation_changes.patch').write_text(''.join(changes))
    print('Wrote lineage for',len(lineage),'objects; source revisions and execution metadata')


if __name__=='__main__':main()
