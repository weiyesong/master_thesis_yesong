"""E: exact conditional Beta-Binomial reference, no sampled training."""
import time
import numpy as np
from scipy.special import digamma
from scipy.stats import binom, beta as beta_dist
from scipy.integrate import quad
from common import OUT, csvout, dump
from metrics import entropy


def moments(alpha,beta):
    s=alpha+beta;m=alpha/s
    tu=entropy(m,True)
    ee=digamma(s+1)-(alpha*digamma(alpha+1)+beta*digamma(beta+1))/s
    return dict(shannon_tu=tu,shannon_ee=ee,shannon_mi=tu-ee,
                gini_tu=2*m*(1-m),gini_ee=2*alpha*beta/(s*(s+1)),
                gini_eu=2*alpha*beta/(s*s*(s+1)))


def main():
    start=time.perf_counter();terms=[];summaries=[];valid=[]
    for a in [.1,.4]:
        for n in [20,200]:
            strata=[]
            for x,p in [(-1,a),(1,1-a)]:
                k=np.arange(n+1);w=binom.pmf(k,n,p);alpha=1+k;beta=1+n-k
                m=moments(alpha,beta);oracle=float(entropy(p,True))
                assert abs(w.sum()-1)<1e-12
                sr=float(np.max(abs(m['shannon_tu']-m['shannon_ee']-m['shannon_mi'])))
                gr=float(np.max(abs(m['gini_tu']-m['gini_ee']-m['gini_eu'])))
                assert max(sr,gr)<1e-12
                summary={'ambiguity':a,'n_per_stratum':n,'x':x,'oracle_entropy':oracle,'oracle_gini':2*p*(1-p),
                         **{name:float(np.dot(w,v)) for name,v in m.items()}}
                summary['ee_minus_oracle']=summary['shannon_ee']-oracle
                summary['expected_absolute_ee_error']=float(np.dot(w,abs(m['shannon_ee']-oracle)))
                summary['expected_squared_ee_error']=float(np.dot(w,(m['shannon_ee']-oracle)**2))
                strata.append(summary)
                valid.append({'a':a,'n':n,'x':x,'weight_sum':float(w.sum()),'shannon_identity_residual':sr,'gini_identity_residual':gr})
                for i in range(n+1):terms.append({'ambiguity':a,'n_per_stratum':n,'x':x,'k':int(k[i]),'binomial_weight':w[i],**{name:v[i] for name,v in m.items()}})
            merged={name:float(np.mean([s[name] for s in strata])) for name in strata[0] if name not in ['x','ambiguity','n_per_stratum']}
            merged.update(ambiguity=a,n_per_stratum=n,total_labels=2*n,scope='X-conditional quantities, then equal X average',status='exploratory')
            summaries.append(merged)
    integration=[]
    for alpha,beta in [(1,1),(3,19),(9,13),(21,181),(81,121)]:
        ee=quad(lambda p:float(entropy(p,True))*beta_dist.pdf(p,alpha,beta),0,1,epsabs=1e-12,epsrel=1e-12)[0]
        ge=quad(lambda p:2*p*(1-p)*beta_dist.pdf(p,alpha,beta),0,1,epsabs=1e-12,epsrel=1e-12)[0]
        m=moments(alpha,beta);err=max(abs(ee-m['shannon_ee']),abs(ge-m['gini_ee']))
        assert err<1e-10
        integration.append({'alpha':alpha,'beta':beta,'numerical_EH':ee,'analytic_EH':float(m['shannon_ee']),'maximum_abs_error':float(err)})
    lookup={(s['ambiguity'],s['n_per_stratum']):s for s in summaries};interactions=[]
    for name in ['oracle_entropy','shannon_tu','shannon_ee','shannon_mi','gini_tu','gini_ee','gini_eu','ee_minus_oracle']:
        value=lookup[(.4,200)][name]-lookup[(.4,20)][name]-lookup[(.1,200)][name]+lookup[(.1,20)][name]
        interactions.append({'metric':name,'difference_in_differences':value,'contrast':'U(.4,200)-U(.4,20)-U(.1,200)+U(.1,20)','interpretation':'nonadditivity of this estimator and scale; not physical source coupling'})
    assert len(terms)==888
    csvout(OUT/'toy/posterior_terms.csv',terms);csvout(OUT/'toy/condition_means.csv',summaries);csvout(OUT/'toy/interactions.csv',interactions)
    dump(OUT/'toy/verification.json',{'terms':len(terms),'conditions':len(summaries),'strata_checks':valid,'quadrature_checks':integration,'seconds':time.perf_counter()-start,'FM_training':0,'model_inference':0,'independent_evidence_warning':'two X strata are mirror constructions, not independent experimental replicates'})
    print('E complete: 888 weighted posterior terms, 4 conditions')


if __name__=='__main__':main()
