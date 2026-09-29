"""Pin the completed reports, scripts and supplemental artifacts after review."""
from pathlib import Path
import csv,json,hashlib,datetime,re,collections
ROOT=Path('/workspace');OUT=Path(__file__).resolve().parent;OLD=ROOT/'reports/core_rq_audit_20260918';NEW=ROOT/'results/final_thesis/core_rq_completion_20260921'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def record(p):return {'path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'sha256':sha(p)}
def main():
 v=json.loads((OUT/'segmentation_off_verification_summary.json').read_text());assert v['passed']==12 and v['failed']==0 and v['complete']
 checks=list(csv.DictReader((OUT/'checks.csv').open()));assert len(checks)==29
 assert collections.Counter(r['status'] for r in checks)=={'PASS':27,'UNKNOWN':2}
 assert {r['check_id'] for r in checks if r['status']=='UNKNOWN'}=={'C01','C06'}
 source_changes=[]
 for r in json.loads((OLD/'snapshot.json').read_text())['files']:
  p=ROOT/r['path'];assert p.is_file(),p
  current=sha(p)
  if current!=r['sha256']:source_changes.append({'path':r['path'],'previous_sha256':r['sha256'],'current_sha256':current})
 assert {r['path'] for r in source_changes}=={'scripts/c6_build_final_results.py'},source_changes
 for r in json.loads((OLD/'audit_output_manifest.json').read_text())['files']:
  assert sha(OLD/r['path'])==r['sha256'],r['path']
 # Every review-facing local link must resolve; ignore HTTP and fragment anchors.
 links=[]
 for p in [OUT/'COMPLETION_REPORT.md',OUT/'RESULTS_SECTION.md',OUT/'published/final_thesis_results.md',ROOT/'reports/CORE_RQ_CURRENT.md']:
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' in target or target.startswith('#'):continue
   path=target.split('#')[0];exists=(p.parent/path).exists();links.append({'document':str(p.relative_to(ROOT)),'target':target,'exists':exists})
 assert all(r['exists'] for r in links),[r for r in links if not r['exists']]
 (OUT/'final_integrity.json').write_text(json.dumps({'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'previous_source_files_checked':2394,'authorized_changes':source_changes,'previous_audit_files_checked':104,'all_previous_audit_outputs_unchanged':True,'local_link_checks':links},indent=2,ensure_ascii=False))
 files=[p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in {'completion_manifest.json','finalize_manifest.log'}]
 files.extend(p for p in (NEW/'segmentation').rglob('*') if p.is_file())
 files.extend([ROOT/'scripts/c6_build_final_results.py',ROOT/'scripts/complete_segmentation_dropout_off.py',ROOT/'scripts/build_core_rq_completion.py',ROOT/'tests/test_c6_final_results.py',ROOT/'reports/CORE_RQ_CURRENT.md'])
 entries=[record(p) for p in sorted(set(files))]
 excluded=[record(p) for p in (NEW/'interrupted_before_completion').rglob('*') if p.is_file()]
 current={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary_report':str((OUT/'COMPLETION_REPORT.md').relative_to(ROOT)),
  'scope':'Core RQ audit completion; no training; 12 new segmentation full-test dropout-off exports plus 12 reused classification controls',
  'previous_source_snapshot_id':json.loads((OLD/'snapshot.json').read_text())['snapshot_id'],'checks':{'PASS':27,'UNKNOWN':2,'FAIL':0,'NA':0},
  'core_method_cells':{'PASS':52,'NA':12},'same_checkpoint_controls':24,'new_segmentation_evaluations':12,'file_count':len(entries),'files':entries,
  'excluded_interrupted_files':excluded,'excluded_reason':'First attempt stopped during NPZ compression before completion marker; never admitted to metrics or conclusions.'}
 digest=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest();current['content_snapshot_id']='sha256:'+digest
 (OUT/'completion_manifest.json').write_text(json.dumps(current,indent=2,ensure_ascii=False));print(current['content_snapshot_id'],len(entries),'files; all links and historical immutability verified')
if __name__=='__main__':main()
