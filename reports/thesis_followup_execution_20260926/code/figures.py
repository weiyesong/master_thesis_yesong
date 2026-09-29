"""Standalone thesis figures, with source tables and explicit evaluation units."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, NullFormatter
from common import OUT,MASTER,get_inputs,dump

COLORS={'deterministic':'#2c6caa','temperature_scaling':'#4a9c5b','mc_dropout':'#e38b22','mc_dropout_off':'#8b79ad','deep_ensemble':'#c94f52'}
LABELS={'deterministic':'D','temperature_scaling':'TS','mc_dropout':'MC (T=30)','mc_dropout_off':'MC Off','deep_ensemble':'DE (M=3)'}
CONFIG_MARKERS={('dofa','frozen'):'o',('dofa','full_finetune'):'^',('panopticon','frozen'):'s',('panopticon','full_finetune'):'D'}


def proxy_legend(fig):
    handles=[Line2D([],[],ls='',marker=marker,color='gray',label=f'{model} / {adapt}') for (model,adapt),marker in CONFIG_MARKERS.items()]
    handles += [Line2D([],[],ls='',marker='o',color=COLORS[method],label=LABELS[method]) for method in ['mc_dropout','deep_ensemble']]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=9)


def main():
    dest=OUT/'figures';dest.mkdir(exist_ok=True);manifest=[]
    def save(fig,name,sources,caption):
        fig.savefig(dest/(name+'.pdf'),bbox_inches='tight');fig.savefig(dest/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
        manifest.append({'figure':name,'sources':sources,'caption':caption})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    curves=pd.read_csv(OUT/'curves.csv');metrics=pd.read_csv(OUT/'metrics.csv');pairs=pd.read_csv(OUT/'paired_effects.csv')
    for ds in ['eurosat','treesatai','cloudsen12','spacenet7']:
        unit={'eurosat':'image_top1','treesatai':'image_hamming','cloudsen12':'image_pixel_error_fraction','spacenet7':'image_pixel_error_fraction'}[ds]
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
        for ax,(model,adapt) in zip(axes.flat,[(m,a) for m in ['dofa','panopticon'] for a in ['frozen','full_finetune']]):
            d=curves[(curves.dataset==ds)&(curves.model==model)&(curves.adaptation==adapt)&(curves.analysis_unit==unit)&(curves.score=='msp')&((curves.seed==42)|curves.seed.isna())]
            for method,g in d.groupby('method'):
                g=g.sort_values('coverage');ax.plot(g.coverage,g.risk,color=COLORS[method],label=LABELS[method],lw=1.6 if method=='mc_dropout_off' else 1.9,ls=':' if method=='mc_dropout_off' else '-',alpha=.9 if method=='mc_dropout_off' else 1)
                if method=='deterministic':ax.axhline(g.random_risk.iloc[0],color=COLORS[method],ls=':',alpha=.7,label='D random ordering')
            ax.set_title(f'{model.upper()} / {adapt.replace("_"," ")}');ax.set_xlabel('Retained image coverage');ax.set_ylabel('Selective risk');ax.grid(alpha=.15)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.035));fig.suptitle(f'{ds}: MSP image rejection\nseed 42 predictors; one three-member ensemble; exploratory');fig.tight_layout(rect=(0,.045,1,.92))
        save(fig,'risk_coverage_'+ds,['curves.csv'],f'{ds}; {unit}; all test images; lower risk is better; curves compare different predictors as well as rankings. MC and MC Off curves can overlap closely; Off is dotted. No training-seed confidence claim.')
    # TS calibration and behavior are distinct axes on identical checkpoints.
    t=pairs[(pairs.contrast=='TS_minus_D')&(pairs.score=='msp')&(pairs.metric=='auroc')].copy();master=pd.DataFrame(MASTER)
    master=master[master.record_type=='INDIVIDUAL_SEED']
    changes=[]
    for _,r in t.iterrows():
        b=master[(master.dataset==r.dataset)&(master.model==r.model)&(master.adaptation==r.adaptation)&(master.seed==str(int(r.seed)))]
        de=float(b[b.uq_method=='temperature_scaling'].ece_15.iloc[0])-float(b[b.uq_method=='deterministic'].ece_15.iloc[0]);changes.append(de)
    t['delta_ece']=changes;t.to_csv(OUT/'TS_calibration_behavior.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,5))
    for (model,adapt),g in t.groupby(['model','adaptation']):
        ax.errorbar(g.delta_ece,g.point_difference,yerr=np.stack([g.point_difference-g.ci_low,g.ci_high-g.point_difference]),fmt='o' if model=='dofa' else 's',capsize=2,label=f'{model} / {adapt}',color='#2c6caa' if adapt=='frozen' else '#d87522',alpha=.85)
        for _,r in g.iterrows():ax.annotate(str(int(r.seed)),(r.delta_ece,r.point_difference),xytext=(4,3),textcoords='offset points',fontsize=8)
    ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='gray',lw=.8);ax.set(xlabel='TS - D: ECE-15 (lower is better)',ylabel='TS - D: MSP error AUROC (higher is better)',title='EuroSAT: calibration and error ranking need not move together');ax.legend(fontsize=8);fig.tight_layout()
    save(fig,'TS_calibration_and_error_ranking',['TS_calibration_behavior.csv','TS_same_checkpoint_verification.csv'],'Twelve same-checkpoint pairs; all argmax decisions unchanged. Bars are paired spatial-group bootstrap intervals for fixed trained models, not across-seed intervals.')
    # Scale diagnostics from saved logits; show all fixed D/TS seed pairs.
    scale=pd.read_csv(OUT/'scale.csv');fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,measure in zip(axes,['centered_logit_norm','top1_top2_logit_margin']):
        d=scale[(scale.dataset=='eurosat')&(scale.measure==measure)&scale.method.isin(['deterministic','temperature_scaling'])]
        for (model,adapt),g in d.groupby(['model','adaptation']):
            pivot=g.pivot(index='seed',columns='method',values='mean')
            ax.scatter(pivot.deterministic,pivot.temperature_scaling,marker='o' if model=='dofa' else 's',color='#2c6caa' if adapt=='frozen' else '#d87522',label=f'{model} / {adapt}')
            for seed,row in pivot.iterrows():ax.annotate(str(int(seed)),(row.deterministic,row.temperature_scaling),xytext=(4,3),textcoords='offset points',fontsize=8)
        high=float(d['mean'].max())*1.08;ax.plot([0,high],[0,high],ls=':',color='gray');ax.set(xlim=(0,high),ylim=(0,high),xlabel='D: test-image mean',ylabel='TS: test-image mean',title=measure.replace('_',' '));ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.suptitle('EuroSAT: saved-logit scale under the existing temperatures\n12 same-checkpoint pairs; descriptive, exploratory');fig.tight_layout(rect=(0,0,1,.90))
    save(fig,'TS_logit_scale',['scale.csv','TS_same_checkpoint_verification.csv'],'Means over 2714 test images per object. Centered norms remove common logit-shift arbitrariness; logit margins are distinct from probability margins. No causal mediation claim.')
    # Fixed-predictor MI vs MSP avoids interpreting a change of error set as a score effect.
    within=pd.read_csv(OUT/'within_predictor_score_effects.csv');fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,ds,unit in zip(axes,['eurosat','treesatai'],['image_top1','micro_label_decision']):
        d=within[(within.dataset==ds)&(within.analysis_unit==unit)&(within.metric=='auroc')&(within.score=='mutual_information')&((within.seed==42)|within.seed.isna())].sort_values(['model','adaptation','method'])
        xx=np.arange(len(d));ax.errorbar(d.point_difference,xx,xerr=np.stack([d.point_difference-d.ci_low,d.ci_high-d.point_difference]),fmt='o',capsize=3,color='#8b4a84');ax.axvline(0,color='gray',lw=.8)
        ax.set_yticks(xx,[f'{r.model[:4]} / {"F" if r.adaptation=="frozen" else "Full"} / {LABELS[r.method]}' for r in d.itertuples()]);ax.set_xlabel('Error AUROC: MI - MSP');ax.set_title(ds+' / '+unit);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Same predictor, two uncertainty scores (seed 42 / one DE group)');fig.tight_layout(rect=(0,0,1,.94))
    save(fig,'MI_vs_MSP_error_ranking',['within_predictor_score_effects.csv'],'Fixed-predictor score contrasts. Conditional group-bootstrap intervals; no universal score ranking is inferred.')
    # Full image-level rejection may disproportionately remove building pixels.
    cc=pd.read_csv(OUT/'full_image_class_coverage.csv');d=cc[(cc.dataset=='spacenet7')&(cc.class_name=='building')&(cc.score=='msp')&(cc.image_coverage==.5)]
    fig,ax=plt.subplots(figsize=(9,4));labels=[f'{r.model[:4]} {"F" if r.adaptation=="frozen" else "Full"} {LABELS[r.method]}' for r in d.itertuples()]
    ax.bar(np.arange(len(d)),d.class_pixel_retention,color=[COLORS[m] for m in d.method]);ax.axhline(.5,color='gray',ls='--',label='50% class retention reference');ax.set_xticks(np.arange(len(d)),labels,rotation=65,ha='right');ax.set(ylabel='Fraction of true building pixels retained',title='SpaceNet7: retain 50% of images using MSP');ax.legend();fig.tight_layout()
    save(fig,'spacenet_building_retention',['full_image_class_coverage.csv'],'Ground truth is used only to diagnose selected image content. Improvement in all-pixel risk is not a building-detection guarantee.')
    # Conditional proxy response in an exactly enumerated known generative model.
    toy=pd.read_csv(OUT/'toy/condition_means.csv');fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,name in zip(axes.flat,['shannon_tu','shannon_ee','shannon_mi','gini_tu','gini_ee','gini_eu']):
        for a,g in toy.groupby('ambiguity'):
            g=g.sort_values('n_per_stratum');color='#2c6caa' if a==.1 else '#d87522'
            ax.plot(g.n_per_stratum,g[name],marker='o',color=color,label=f'a={a:g}')
            if name=='shannon_ee':ax.axhline(g.oracle_entropy.iloc[0],color=color,ls=':',alpha=.8)
            if name=='gini_ee':ax.axhline(g.oracle_gini.iloc[0],color=color,ls=':',alpha=.8)
        ax.set(xscale='log',xticks=[20,200],xticklabels=['20','200'],xlabel='Labels per X stratum',ylabel='nats' if name.startswith('shannon') else 'binary categorical Gini',title=name.replace('_',' ').upper());ax.grid(alpha=.15)
        ax.xaxis.set_minor_locator(NullLocator());ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0,0].legend();fig.suptitle('Analytic reference: conditional quantities, then equal average over X\nDotted EE references use the known generating distribution; no FM training');fig.tight_layout(rect=(0,0,1,.9))
    save(fig,'analytic_proxy_reference',['toy/condition_means.csv','toy/verification.json'],'Exact binomial-weighted posterior expectations under Beta(1,1); four conditions, two mirror X strata. Nonadditivity does not identify physical AU/EU coupling.')
    # Per-method proxy means for classification; no crossing task units.
    pm=pd.read_csv(OUT/'proxy_means.csv');fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,ds in zip(axes,['eurosat','treesatai']):
        d=pm[(pm.dataset==ds)&pm.method.isin(['mc_dropout','deep_ensemble'])&((pm.seed==42)|pm.seed.isna())]
        pivot=d.pivot(index=['model','adaptation','method'],columns='score',values='mean').reset_index()
        for r in pivot.itertuples():ax.scatter(r.expected_entropy,r.mutual_information,color=COLORS[r.method],marker=CONFIG_MARKERS[(r.model,r.adaptation)],s=58,edgecolors='white',linewidths=.4)
        ax.set(xlabel='Mean expected entropy',ylabel='Mean MI-style disagreement',title=ds+(' (categorical nats)' if ds=='eurosat' else ' (mean Bernoulli nats)'));ax.grid(alpha=.15)
    fig.suptitle('Saved-prediction proxies; seed 42 / one DE group; exploratory');proxy_legend(fig);fig.tight_layout(rect=(0,.16,1,.93))
    save(fig,'classification_proxy_means',['proxy_means.csv'],'MC T=30 and DE M=3 describe different empirical prediction distributions. Separate panels have different task semantics.')
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,ds in enumerate(['eurosat','treesatai']):
        d=pm[(pm.dataset==ds)&pm.method.isin(['mc_dropout','deep_ensemble'])&((pm.seed==42)|pm.seed.isna())]
        pivot=d.pivot(index=['model','adaptation','method'],columns='score',values='mean').reset_index()
        for col,(shannon,gini,title) in enumerate([('entropy','gini_total','Total'),('expected_entropy','gini_expected','Expected component'),('mutual_information','gini_disagreement','Disagreement component')]):
            ax=axes[row,col]
            for r in pivot.to_dict('records'):
                ax.scatter(r[shannon],r[gini],color=COLORS[r['method']],marker=CONFIG_MARKERS[(r['model'],r['adaptation'])],s=58,edgecolors='white',linewidths=.4)
            ax.set(xlabel='Shannon (nats)',ylabel='Categorical Gini' if ds=='eurosat' else 'Mean Bernoulli Gini',title=ds+' / '+title);ax.grid(alpha=.15)
    fig.suptitle('Different functionals of the same saved prediction sets\nseed 42 / one DE group; exploratory');proxy_legend(fig);fig.tight_layout(rect=(0,.09,1,.90))
    save(fig,'Shannon_Gini_classification',['proxy_means.csv'],'Full-test sample means: EuroSAT 2714 images; TreeSat 2000 images, mean over 15 Bernoulli labels. Axes have different scales/units; neither agreement nor differences identify physical uncertainty sources.')
    dump(dest/'figure_manifest.json',manifest)
    print('Created',len(manifest),'standalone figures (PDF + PNG)')


if __name__=='__main__':main()
