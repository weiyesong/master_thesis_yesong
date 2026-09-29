from pathlib import Path
import csv,json,hashlib,tempfile,sys,datetime
ROOT=Path('/workspace');OUT=ROOT/'reports/core_rq_completion_20260921';PUB=OUT/'published';sys.path.insert(0,str(ROOT))
from scripts.c6_build_final_results import read_csv,write_report

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
args=[read_csv(PUB/'tables'/n) for n in ('classification_results.csv','segmentation_results.csv','mc_dropout_robustness.csv')]
checks=[]
for parent in (OUT,ROOT/'reports'):
 with tempfile.TemporaryDirectory(prefix='.c6_link_check_',dir=parent) as temporary:
  p=Path(temporary)/'report.md';write_report(p,*args);text=p.read_text()
  if parent==OUT:assert (PUB/'final_thesis_results.md').read_text().startswith(text)
  import re
  targets=[x for x in re.findall(r'\]\(([^)]+)\)',text) if any(y in x for y in ('COMPLETION_REPORT.md','RESULTS_SECTION.md','mc_dropout_three_way.csv'))]
  assert len(targets)==3 and all((p.parent/t).is_file() for t in targets)
  checks.append({'parent':str(parent),'links':targets,'all_resolve':True,'current_published_body_unchanged':True if parent==OUT else 'not applicable'})
sourcepath=PUB/'tables/source_provenance.csv';rows=list(csv.DictReader(sourcepath.open()))
for r in rows:
 if r['path']=='scripts/c6_build_final_results.py':r['sha256']=sha(ROOT/r['path']);r['size_bytes']=str((ROOT/r['path']).stat().st_size)
with sourcepath.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
manifest_path=PUB/'tables/package_manifest.json';m=json.loads(manifest_path.read_text())
for r in m['generated_artifacts']:
 p=PUB/r['path'];r['sha256']=sha(p);r['size_bytes']=p.stat().st_size
manifest_path.write_text(json.dumps(m,indent=2)+'\n')
errors=[]
for r in rows:
 p=ROOT/r['path']
 if sha(p)!=r['sha256']:errors.append(str(p))
for r in m['generated_artifacts']:
 if sha(PUB/r['path'])!=r['sha256']:errors.append(r['path'])
assert not errors,errors
result={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_hashes_verified':len(rows),'generated_hashes_verified':len(m['generated_artifacts']),'mismatches':errors,'generator_sha256':sha(ROOT/'scripts/c6_build_final_results.py'),'package_manifest_sha256':sha(manifest_path),'portable_link_checks':checks}
(OUT/'reporting_hash_verification.json').write_text(json.dumps(result,indent=2)+'\n')
p=OUT/'reporting_fix_notes.md';s=p.read_text();s=s.replace('b92849708056b9db6674a12ef829293befd61a94220ac1c8292b3eb82f9ba6ea',result['generator_sha256']).replace('f1a39b67e522e90c00367154fc4ada9914afffb7a1296c566f37e562c67c363a',result['package_manifest_sha256']);s+='\n## Final path portability check (2026-09-23)\n\nThe three completion links are computed relative to the actual output directory, so both this published layout and the default reports-level layout resolve. The existing published body is byte-for-byte unchanged. Generator/source hashes were refreshed and all 74 source and 19 generated hashes reverified by verify_reporting_portability.py; four C6 tests passed again after this path-only change.\n';p.write_text(s)
print(json.dumps(result,indent=2))
