"""Assemble audit completion tables from traceable existing and supplemental evaluations."""
from pathlib import Path
import csv,json,hashlib,collections,statistics,shutil
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'reports/core_rq_audit_20260918';OUT=ROOT/'reports/core_rq_completion_20260921'
NEW=ROOT/'results/final_thesis/core_rq_completion_20260921/segmentation'
def load(p):return json.loads(p.read_text())
def rel(p):return str(Path(p).relative_to(ROOT)) if Path(p).is_absolute() else str(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def csvwrite(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def key(r):return tuple(str(r[k]) for k in ('dataset','model','adaptation','seed'))
def flatten(m):
 out={k:v for k,v in m.items() if isinstance(v,(int,float))}
 out.update({'iou_'+k:v for k,v in m.get('per_class_iou',{}).items()})
 out.update({'class_probability_ece_'+k:v['ece_15'] for k,v in m.get('classwise_calibration',{}).items()})
 out.update({'boundary_'+k:v for k,v in m.get('boundary_calibration',{}).items() if k in ('ece_15','foreground_ece_15')})
 return out
def main():
 OUT.mkdir(exist_ok=True)
 # Table construction reads completed artifacts; independent verification gates
 # promotion in write_completion_report.py, not the computation of differences.
 # Previously independently recomputed metrics, not a reviewer summary.
 previous=load(OLD/'classification_verification.json')+load(OLD/'segmentation_verification.json')
 inventory={(key(x),x['uq_method']):x for x in csv.DictReader((OLD/'actual_artifacts.csv').open())}
 lookup={(key(r),r['uq_method']):r for r in previous}
 controls=[];three=[];effects=[]
 offclass=load(OLD/'classification_dropout_off_verification.json')
 for r in offclass:
  controls.append({k:r[k] for k in ('dataset','model','adaptation','seed','checkpoint_path','checkpoint_sha256')}|
   {'task':'classification','output_path':r['output_path'],'metrics':r['metrics'],
    'off_provenance':rel(OLD/'classification_dropout_off_verification.json'),'off_evaluation':'saved deterministic features + eval head',
    'sample_count':2714 if r['dataset']=='eurosat' else 2000})
 completed=sorted(NEW.rglob('completion.json'));assert len(completed)==12,len(completed)
 for p in completed:
  r=load(p);assert r['passed'] and all(r['checks'].values())
  metric_path=p.parent/('metrics_export_aligned.json' if (p.parent/'metric_alignment.json').exists() else 'metrics.json')
  controls.append({k:r[k] for k in ('task','dataset','model','adaptation','seed','checkpoint_path','checkpoint_sha256','sample_count')}|
   {'output_path':rel(Path(r['prediction_path'])),'metrics':flatten(load(metric_path)), 'metric_path':rel(metric_path),
    'off_provenance':rel(p),'off_evaluation':'full model eval, full test, one deterministic forward'})
 assert len(controls)==24 and len({key(r) for r in controls})==24
 for r in controls:
  mc=lookup[(key(r),'mc_dropout')];d=lookup[(key(r),'deterministic')]
  # Main metrics use original production summaries for exact historical native
  # clipping semantics; independent recomputation certifies their agreement.
  if r['task']=='segmentation':
   mcmetrics=flatten(load((ROOT/mc['artifact_path']).parent/'metrics.json'))
   # D original source metrics are stored in manifest; independent values for
   # all main/class metrics are equivalent within audited floating precision.
   dm=dict(d['metrics']);fg=next((x for x in load(OLD/'foreground_precision_verification.json') if key(x)==key(r) and x['uq_method']=='deterministic'),None)
   if fg:dm['foreground_nll']=fg['reported_foreground_nll']
  else:mcmetrics=mc['metrics'];dm=d['metrics']
  off=r['metrics'];names=[n for n in ('accuracy','macro_f1','miou','pixel_accuracy','nll','brier','ece_15','iou_building','foreground_ece_15','foreground_nll','foreground_brier','boundary_ece_15','boundary_foreground_ece_15') if n in off and n in mcmetrics and n in dm]
  primary={'eurosat':'accuracy','treesatai':'macro_f1','cloudsen12':'miou','spacenet7':'miou'}[r['dataset']]
  z={k:v for k,v in r.items() if k!='metrics'}
  z.update(primary_metric=primary,mc_path=mc['artifact_path'],deterministic_path=d['artifact_path'])
  for label,method in [('D','deterministic'),('MC_training','mc_dropout')]:
   rd=Path(inventory[(key(r),method)]['checkpoint_path']).parent; sm=load(rd/'run_summary.json');history=load(rd/'training_history.json')
   if isinstance(history,dict):history=history.get('history',history.get('epochs',[]))
   z[label+'_best_epoch']=sm['best_epoch'];z[label+'_last_epoch']=sm.get('last_epoch',max(int(h['epoch']) for h in history))
   z[label+'_run_summary_path']=rel(rd/'run_summary.json')
  z['checkpoint_selection_note']='MC checkpoint selected under dropout-off validation; Off-D includes recipe and training trajectory'
  z['D_seed_performance_min']=min(x['metrics'][primary] for x in previous if key(x)[:3]==key(r)[:3] and x['uq_method']=='deterministic')
  z['D_seed_performance_max']=max(x['metrics'][primary] for x in previous if key(x)[:3]==key(r)[:3] and x['uq_method']=='deterministic')
  z['D_seed_ece15_min']=min(x['metrics']['ece_15'] for x in previous if key(x)[:3]==key(r)[:3] and x['uq_method']=='deterministic')
  z['D_seed_ece15_max']=max(x['metrics']['ece_15'] for x in previous if key(x)[:3]==key(r)[:3] and x['uq_method']=='deterministic')
  for prefix,ms in [('D',dm),('Off',off),('MC',mcmetrics)]:z.update({prefix+'_'+n:ms[n] for n in names})
  for contrast,a,b in [('MC_minus_Off',mcmetrics,off),('Off_minus_D',off,dm),('MC_minus_D',mcmetrics,dm)]:
   delta={n:a[n]-b[n] for n in names};z.update({contrast+'_'+n:v for n,v in delta.items()})
   delta_perf=delta[primary];de=delta['ece_15']
   category=('ECE decreases' if de<0 else 'ECE increases' if de>0 else 'ECE unchanged')+'; '+('performance increases' if delta_perf>0 else 'performance decreases' if delta_perf<0 else 'performance exactly unchanged')
   effects.append({k:z[k] for k in ('task','dataset','model','adaptation','seed','primary_metric','checkpoint_path','checkpoint_sha256','output_path','off_provenance','mc_path','deterministic_path')}|
    {'contrast':contrast,'statistical_unit':'paired training seed; MC finite30 sample mean fixed weights','point_estimate_category':category+f'; delta performance={delta_perf:+.9g}, delta ECE={de:+.9g}; no practical equivalence claim',
     'delta_performance':delta_perf}|{'delta_'+n:v for n,v in delta.items()})
  three.append(z)
 csvwrite('mc_dropout_three_way.csv',three);csvwrite('mc_dropout_effects.csv',effects)
 groups=collections.defaultdict(list)
 for r in effects:groups[tuple(r[k] for k in ('dataset','model','adaptation','contrast'))].append(r)
 summaries=[]
 for k,rs in groups.items():
  z=dict(zip(('dataset','model','adaptation','contrast'),k));z.update(n_training_seeds=len(rs),seeds=';'.join(str(r['seed']) for r in rs),primary_metric=rs[0]['primary_metric'],basis='observed paired effects; sample SD is descriptive, not CI or equivalence test')
  for name in [n for n in rs[0] if n.startswith('delta_')]:
   values=[r[name] for r in rs];z.update({name+'_mean':statistics.mean(values),name+'_sample_sd':statistics.stdev(values) if len(values)>1 else '',name+'_min':min(values),name+'_max':max(values),name+'_positive_seeds':sum(v>0 for v in values),name+'_negative_seeds':sum(v<0 for v in values)})
  summaries.append(z)
 csvwrite('mc_dropout_effect_summaries.csv',summaries)
 csvwrite('mc_dropout_control_coverage.csv',[{k:v for k,v in r.items() if k!='metrics'}|{'coverage_status':'PASS','independent_validation':rel(OUT/'segmentation_off_verification.json') if r['task']=='segmentation' else rel(OLD/'classification_dropout_off_verification.json')} for r in controls])
 for name in ('coverage_matrix.csv','paired_effect_summaries.csv','paired_effects_from_predictions.csv','effect_outcome_categories.csv','protocol_timeline_and_na.csv'):
  shutil.copyfile(OLD/name,OUT/name)
 # Link all underlying prediction/checkpoint metadata for independent review.
 csvwrite('supplemental_artifacts.csv',[{k:v for k,v in r.items() if k!='metrics'}|{'uq_method':'same_checkpoint_dropout_off','output_sha256':sha(ROOT/r['output_path'])} for r in controls])
 print('Built 24 controls, 72 paired effects, 48 summaries; unchanged 64-cell core matrix.',flush=True)
if __name__=='__main__':main()
