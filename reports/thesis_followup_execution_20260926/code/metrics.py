"""Checkpoint-free scientific measures. Larger scores mean more uncertain."""
from __future__ import annotations
import numpy as np
from scipy.special import xlogy

COVERAGES = (1., .9, .8, .5)


def entropy(p, binary=False):
    p = np.asarray(p, dtype=np.float64)
    if binary:
        return -xlogy(p, p) - xlogy(1-p, 1-p)
    return -xlogy(p, p).sum(axis=-1)


def decomposition(draws, binary=False):
    """[N,draw,C] or [pixels,draw,C]; Bernoulli output retains C labels."""
    p = np.asarray(draws, dtype=np.float64)
    mean = p.mean(axis=1)
    if binary:
        tu = entropy(mean, True)
        ee = entropy(p, True).mean(axis=1)
        gt = mean*(1-mean)
        ge = (p*(1-p)).mean(axis=1)
        gu = p.var(axis=1, ddof=0)
    else:
        tu = entropy(mean)
        ee = entropy(p).mean(axis=1)
        gt = 1-(mean**2).sum(axis=-1)
        ge = (1-(p**2).sum(axis=-1)).mean(axis=1)
        gu = p.var(axis=1, ddof=0).sum(axis=-1)
    return mean, dict(entropy=tu, expected_entropy=ee, mutual_information=tu-ee,
                      gini_total=gt, gini_expected=ge, gini_disagreement=gu)


def base_scores(p, binary=False):
    p = np.asarray(p, dtype=np.float64)
    if binary:
        return dict(msp=1-np.maximum(p, 1-p), entropy=entropy(p, True),
                    negative_margin=-np.abs(2*p-1), gini_total=p*(1-p))
    two = np.partition(p, -2, axis=-1)[..., -2:]
    return dict(msp=1-p.max(axis=-1), entropy=entropy(p),
                negative_margin=two[..., 0]-two[..., 1], gini_total=1-(p*p).sum(axis=-1))


def align_indices(ids, reference):
    ids = np.asarray(ids).astype(str)
    reference = np.asarray(reference).astype(str)
    if len(set(ids)) != len(ids) or len(set(reference)) != len(reference):
        raise ValueError('duplicate sample IDs')
    if set(ids) != set(reference):
        raise ValueError('sample ID sets differ')
    index = {v:i for i,v in enumerate(ids)}
    return np.array([index[v] for v in reference], dtype=np.int64)


class RankPlan:
    """Exact tie-aware binary rankings and expected random-within-tie RC.

    Bootstrap weights are nonnegative integer multiplicities, not survey weights.
    Discrete AURC averages expected selective risk at k/N, k=1..N.
    """
    def __init__(self, uncertainty, loss):
        score = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
        self.loss = np.asarray(loss, dtype=np.float64).reshape(-1)
        if score.shape != self.loss.shape or not len(score):
            raise ValueError('score/loss must have equal nonempty shape')
        if not np.isfinite(score).all() or not np.isfinite(self.loss).all():
            raise ValueError('nonfinite score/loss')
        if self.loss.min() < 0 or self.loss.max() > 1:
            raise ValueError('loss outside [0,1]')
        self.order = np.argsort(score, kind='stable')
        self.sorted_loss = self.loss[self.order]
        ss = score[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ss) != 0)+1]
        self.binary = bool(np.isin(self.loss, [0,1]).all())
        self.n = len(score)

    def evaluate(self, weights=None, coverages=COVERAGES):
        if weights is None:
            weights = np.ones((1, self.n), dtype=np.int64)
        w = np.asarray(weights)
        if w.ndim == 1:
            w = w[None]
        if w.shape[1] != self.n or np.any(w<0) or not np.equal(w, np.floor(w)).all():
            raise ValueError('bootstrap weights must be nonnegative integer counts')
        sorted_w = w[:,self.order]
        count = np.add.reduceat(sorted_w, self.starts, axis=1).astype(np.int64)
        mass = np.add.reduceat(sorted_w*self.sorted_loss, self.starts, axis=1)
        cum_n = np.cumsum(count, axis=1)
        cum_e = np.cumsum(mass, axis=1)
        n = cum_n[:,-1]
        e = cum_e[:,-1]
        if np.any(n == 0):
            raise ValueError('empty bootstrap draw')
        prev_n = cum_n-count
        prev_e = cum_e-mass
        avg = np.divide(mass, count, out=np.zeros_like(mass), where=count>0)
        h = np.r_[0., np.cumsum(1/np.arange(1, int(n.max())+1, dtype=np.float64))]
        area = np.sum(avg*count + (prev_e-avg*prev_n)*(h[cum_n]-h[prev_n]), axis=1)/n
        out = dict(base_risk=e/n, aurc=area)
        if self.binary:
            neg = count-mass
            below_neg = prev_n-prev_e
            denominator = e*(n-e)
            numerator = (mass*(below_neg+.5*neg)).sum(axis=1)
            auc = np.divide(numerator, denominator, out=np.full_like(e,np.nan), where=denominator>0)
            high_e = e[:,None]-prev_e
            high_n = n[:,None]-prev_n
            precision = np.divide(high_e, high_n, out=np.zeros_like(high_e), where=high_n>0)
            ap_num = (mass*precision).sum(axis=1)
            ap = np.divide(ap_num,e,out=np.full_like(e,np.nan),where=(e>0)&(e<n))
            out.update(auroc=auc, average_precision=ap)
        else:
            out.update(auroc=np.full(len(n),np.nan), average_precision=np.full(len(n),np.nan))
        for q in coverages:
            target = q*n
            idx = np.minimum(np.sum(cum_n<target[:,None],axis=1),count.shape[1]-1)
            b = np.arange(len(n))
            selected_mass = prev_e[b,idx] + (target-prev_n[b,idx])*avg[b,idx]
            out[f'risk_at_{q:g}'] = selected_mass/target
        return out

    def acceptance(self, coverage):
        """Label-independent fractional tie membership in original sample order."""
        count = np.diff(np.r_[self.starts,self.n])
        previous = np.r_[0,np.cumsum(count)[:-1]]
        frac = np.clip((coverage*self.n-previous)/count,0,1)
        out = np.empty(self.n)
        out[self.order] = np.repeat(frac,count)
        return out


def interval(x):
    a = np.asarray(x,dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan,np.nan,0
    return *np.quantile(a,[.025,.975]),len(a)


def bootstrap_group_counts(groups, replicates=1000, seed=20260926):
    groups = np.asarray(groups).astype(str)
    unique,inverse = np.unique(groups,return_inverse=True)
    rng = np.random.default_rng(seed)
    # Same group universe and seed -> same draws for all matched predictors.
    counts = rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=replicates)
    return counts[:,inverse].astype(np.int32), unique
