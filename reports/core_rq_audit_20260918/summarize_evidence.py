"""Build paired effects and diagnostics exclusively from audit recomputations."""
from pathlib import Path
import csv,json,itertools,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/workspace');OUT=Path(__file__).resolve().parent
data=[];bins=[]
for task in ('classification','segmentation'):
 data+=json.loads((OUT/(task+'_verification.json')).read_text())
 bins+=list(csv.DictReader((OUT/(task+'_reliability_bins.csv')).open()))
assert len(data)==100,len(data)
def key(r):return tuple(r[k] for k in ('dataset','model','adaptation','uq_method','seed'))
index={key(r):r for r in data}
def csvout(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
metrics=['accuracy','macro_f1','miou','pixel_accuracy','iou_building','nll','brier','ece_10','ece_15','ece_30','foreground_ece_15','boundary_ece_15']
dslist=('eurosat','treesatai','cloudsen12','spacenet7');models=('dofa','panopticon');ads=('frozen','full_finetune')
effects=[]
def effect(rq,base,comp,seed,basis,extra=None):
 row={'rq':rq,'dataset':comp['dataset'],'model':comp['model'],'adaptation':comp['adaptation'],'method':comp['uq_method'],'seed':seed,'basis':basis,'baseline_artifact_paths':';'.join(r['artifact_path'] for r in base),'comparison_artifact_path':comp['artifact_path']}
 if extra:row.update(extra)
 for m in metrics:
  if all(m in r['metrics'] for r in base) and m in comp['metrics']:
   row['baseline_'+m]=float(np.mean([r['metrics'][m] for r in base]));row['comparison_'+m]=comp['metrics'][m];row['delta_'+m]=comp['metrics'][m]-row['baseline_'+m]
 effects.append(row)
for ds,ad,seed in itertools.product(dslist,ads,('42','43','44')):
 effect('RQ1',[index[(ds,'dofa',ad,'deterministic',seed)]],index[(ds,'panopticon',ad,'deterministic',seed)],seed,'Panopticon minus DOFA; entire configuration contrast',{'model':'panopticon_minus_dofa','confound':'backend normalization differs' if ds=='eurosat' else 'within-dataset common input protocol'})
for ds,mo,seed in itertools.product(dslist,models,('42','43','44')):
 effect('RQ2',[index[(ds,mo,'frozen','deterministic',seed)]],index[(ds,mo,'full_finetune','deterministic',seed)],seed,'full minus frozen; adaptation recipes including LR/budget')
for r in data:
 if r['uq_method'] in ('temperature_scaling','mc_dropout'):
  effect('RQ3',[index[(r['dataset'],r['model'],r['adaptation'],'deterministic',r['seed'])]],r,r['seed'],'same checkpoint TS' if r['uq_method']=='temperature_scaling' else 'separate dropout-trained model vs original p=0 model; dropout-off contrast missing')
 elif r['uq_method']=='deep_ensemble':
  effect('RQ3',[index[(r['dataset'],r['model'],r['adaptation'],'deterministic',s)] for s in ('42','43','44')],r,'ensemble(42,43,44)','one ensemble minus mean of its exact 3 members; no across-ensemble SD')
csvout('paired_effects_from_predictions.csv',effects)
groups=collections.defaultdict(list)
for r in effects:groups[tuple(r[k] for k in ('rq','dataset','model','adaptation','method'))].append(r)
summary=[]
for key0,group in groups.items():
 row=dict(zip(('rq','dataset','model','adaptation','method'),key0));row.update(n_contrasts=len(group),seeds=';'.join(r['seed'] for r in group),basis=group[0]['basis'])
 for m in metrics:
  if all('delta_'+m in r for r in group):
   x=np.array([r['delta_'+m] for r in group]);row['delta_'+m+'_mean']=float(x.mean());row['delta_'+m+'_std']=float(x.std(ddof=1)) if len(x)>1 else '';row['delta_'+m+'_min']=float(x.min());row['delta_'+m+'_max']=float(x.max())
 summary.append(row)
csvout('paired_effect_summaries.csv',summary)
# Check existing seed-level arithmetic and mean/sample-SD, not only copied tables.
master=list(csv.DictReader((ROOT/'reports/thesis_master_results.csv').open()));checks=[]
for r in master:
 if r['record_type']!='MEAN_STD':continue
 group=[x for x in data if all(x[k]==r[k] for k in ('dataset','model','adaptation','uq_method'))]
 for m in metrics:
  if r.get(m) and all(m in x['metrics'] for x in group):
   vals=np.array([x['metrics'][m] for x in group]);checks.append(dict(source='reports/thesis_master_results.csv',dataset=r['dataset'],model=r['model'],adaptation=r['adaptation'],method=r['uq_method'],metric=m,mean_difference=float(vals.mean()-float(r[m])),sample_std_difference=float(vals.std(ddof=1)-float(r[m+'_std']))))
for name,rq in [('rq2_adaptation_effects.csv','RQ2'),('rq3_uq_effects.csv','RQ3')]:
 for r in csv.DictReader((ROOT/'reports'/name).open()):
  if r['record_type'] not in ('PAIRED_SEED','AGGREGATED_COMPARISON','PAIRED_MEAN_STD'):continue
  candidates=[e for e in effects if e['rq']==rq and all(e[k]==r[k] for k in ('dataset','model')) and (rq=='RQ2' or (e['adaptation']==r['adaptation'] and e['method']==r['comparison_method'])) and (r['record_type']=='PAIRED_MEAN_STD' or e['seed']==r['seed'] or e['seed'].startswith('ensemble'))]
  if r['record_type']=='PAIRED_MEAN_STD':
   assert len(candidates)==3,(name,r['dataset'],r['model'],r['adaptation'],len(candidates))
   for m in metrics:
    if r.get('delta_'+m) and all('delta_'+m in e for e in candidates):
     values=np.array([e['delta_'+m] for e in candidates]);checks.append(dict(source='reports/'+name,dataset=r['dataset'],model=r['model'],adaptation=r['adaptation'],method=r['comparison_method'],record_type='PAIRED_MEAN_STD',metric=m,delta_mean_difference=float(values.mean()-float(r['delta_'+m])),delta_sample_std_difference=float(values.std(ddof=1)-float(r['delta_'+m+'_std']))))
   continue
  if len(candidates)!=1:checks.append({'source':name,'error':'pair match count '+str(len(candidates)),'row':str(r)});continue
  e=candidates[0]
  for m in metrics:
   if r.get('delta_'+m) and 'delta_'+m in e:checks.append(dict(source='reports/'+name,dataset=r['dataset'],model=r['model'],adaptation=e['adaptation'],method=e['method'],seed=r['seed'],metric=m,delta_difference=e['delta_'+m]-float(r['delta_'+m])))
(OUT/'summary_arithmetic_checks.json').write_text(json.dumps(checks,indent=2))
# Recomputed reliability diagrams with visible bin counts, including empty bins.
def plot_grid(filename,selected,kinds,title):
 fig=plt.figure(figsize=(max(6,4*len(kinds)),4.4*len(selected)+.5))
 gs=fig.add_gridspec(len(selected),len(kinds),hspace=.48,wspace=.28)
 for row_i,cell in enumerate(selected):
  for col_i,kind in enumerate(kinds):
   bb=[r for r in bins if tuple(r[k] for k in ('dataset','model','adaptation'))==cell and r['uq_method']=='deterministic' and r['seed']=='42' and r['kind']==kind and r['n_bins']=='15']
   sub=gs[row_i,col_i].subgridspec(2,1,height_ratios=[3,1],hspace=.12)
   ax=fig.add_subplot(sub[0]);hist=fig.add_subplot(sub[1],sharex=ax)
   nonempty=[r for r in bb if int(r['count'])]
   ax.plot([0,1],[0,1],'--',color='.5',lw=1)
   ax.plot([float(r['mean_confidence']) for r in nonempty],[float(r['observed_frequency']) for r in nonempty],'o-',lw=1.2,ms=3)
   ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2)
   ax.set_title('/'.join(cell[1:])+'\n'+kind,fontsize=9)
   ax.tick_params(labelbottom=False)
   if col_i==0:ax.set_ylabel('Observed frequency',fontsize=9)
   hist.bar([(int(r['bin'])+.5)/15 for r in bb],[int(r['count']) for r in bb],width=.058)
   hist.ticklabel_format(axis='y',style='sci',scilimits=(0,0));hist.set_ylabel('Count',fontsize=8)
   if row_i==len(selected)-1:hist.set_xlabel('Confidence / class probability',fontsize=9)
 fig.suptitle(title,fontsize=12,y=.985);fig.subplots_adjust(top=.94 if len(selected)>1 else .8)
 fig.savefig(OUT/filename,dpi=140,bbox_inches='tight');plt.close(fig)
for ds in dslist:
 cells=[(ds,mo,ad) for mo,ad in itertools.product(models,ads)]
 plot_grid(ds+'_reliability_with_counts.png',cells,['decision_15'],f'{ds}: deterministic seed 42, decision ECE bins (all seeds in CSV)')
 if ds in ('cloudsen12','spacenet7'):
  plot_grid(ds+'_class_probability_reliability.png',cells,[f'class_probability_{i}' for i in range(4 if ds=='cloudsen12' else 2)],f'{ds}: one-vs-rest class probability, all valid pixels; seed 42')
# TreeSatAI has 15 label-probability distributions, report them without pooling.
for mo in models:
 for ad in ads:
  cells=[('treesatai',mo,ad)]
  for block in range(3):plot_grid(f'treesatai_{mo}_{ad}_labels_{block*5}_{block*5+4}.png',cells,[f'class_probability_{i}' for i in range(block*5,block*5+5)],f'TreeSatAI {mo}/{ad}: positive-label probability; seed 42')
print('effects',len(effects),'summaries',len(summary),'arithmetic checks',len(checks))
print('max arithmetic error',max(abs(v) for r in checks for k,v in r.items() if k.endswith('difference')))
