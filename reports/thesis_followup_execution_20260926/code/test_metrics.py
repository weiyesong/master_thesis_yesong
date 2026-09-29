import itertools
import unittest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import roc_auc_score, average_precision_score
from metrics import RankPlan, decomposition, align_indices, base_scores, bootstrap_group_counts


class ScientificTests(unittest.TestCase):
    def test_perfect_reverse_and_ties(self):
        y=np.array([0,0,1,1.])
        self.assertEqual(RankPlan(y,y).evaluate()['auroc'][0],1.)
        self.assertEqual(RankPlan(-y,y).evaluate()['auroc'][0],0.)
        m=RankPlan(np.ones(4),y).evaluate()
        self.assertEqual(m['auroc'][0],.5)
        self.assertEqual(m['average_precision'][0],.5)
        self.assertEqual(m['aurc'][0],.5)

    def test_tie_area_matches_exhaustive_random_order(self):
        score=np.array([0,1,1,1,2.]);loss=np.array([0,.1,.8,1.,1])
        areas=[]
        for perm in itertools.permutations([1,2,3]):
            order=[0,*perm,4]
            areas.append(np.mean(np.cumsum(loss[order])/np.arange(1,6)))
        assert_allclose(RankPlan(score,loss).evaluate()['aurc'][0],np.mean(areas),atol=1e-14)

    def test_weighted_matches_expanded_and_sklearn(self):
        score=np.array([.1,.2,.2,.5,.9]);y=np.array([0,1,0,1,1.]);w=np.array([3,0,2,4,1])
        a=RankPlan(score,y).evaluate(w)
        ss=np.repeat(score,w); yy=np.repeat(y,w)
        b=RankPlan(ss,yy).evaluate()
        for k in a:assert_allclose(a[k],b[k],atol=2e-14)
        assert_allclose(a['auroc'][0],roc_auc_score(yy,ss))
        assert_allclose(a['average_precision'][0],average_precision_score(yy,ss))

    def test_constant_target_is_explicit_na(self):
        for y in [np.zeros(3),np.ones(3)]:
            m=RankPlan([0,1,2],y).evaluate()
            self.assertTrue(np.isnan(m['auroc'][0]))
            self.assertTrue(np.isnan(m['average_precision'][0]))

    def test_decomposition_and_multilabel(self):
        p=np.array([[[.2,.8],[.6,.4]],[[.1,.9],[.7,.3]]])
        for binary in [False,True]:
            mean,d=decomposition(p,binary)
            assert_allclose(d['entropy'],d['expected_entropy']+d['mutual_information'])
            assert_allclose(d['gini_total'],d['gini_expected']+d['gini_disagreement'],atol=1e-15)
        p=np.array([[.8,.7,.1],[.2,.6,.9]])
        y=np.array([[1,0,0],[0,1,1]])
        err=((p>=.5)!=y)
        assert_allclose(err.mean(axis=1),[1/3,0])
        self.assertEqual(base_scores(p,True)['msp'].shape,(2,3))

    def test_id_mismatch_and_permutation(self):
        assert_allclose(align_indices(['b','a'],['a','b']),[1,0])
        with self.assertRaises(ValueError):align_indices(['a','c'],['a','b'])
        with self.assertRaises(ValueError):align_indices(['a','a'],['a','b'])

    def test_centered_logit_gauge_and_ignore_mask(self):
        z=np.array([[1.,4.,-1.],[2.,-2.,3.]])
        shifted=z+np.array([[9.],[-10.]])
        assert_allclose(z-z.mean(axis=1,keepdims=True),shifted-shifted.mean(axis=1,keepdims=True))
        label=np.array([0,1,255,1]);pred=np.array([0,0,0,1]);valid=label!=255
        self.assertEqual(np.mean(pred[valid]!=label[valid]),1/3)
        self.assertFalse(np.any(label[valid]==2))

    def test_group_repeats_share_multiplicity(self):
        w,g=bootstrap_group_counts(['x','x','y'],20,1)
        assert_allclose(w[:,0],w[:,1])
        assert_allclose(w[:,0]+w[:,2],2)

    def test_fractional_workpoint_preserves_expected_tie_risk(self):
        p=RankPlan([0,1,1,1,2,3],[0,0,1,1,1,1])
        assert_allclose(p.acceptance(.5),[1,2/3,2/3,2/3,0,0])
        assert_allclose(p.evaluate(coverages=[.5])['risk_at_0.5'],[4/9])

    def test_boolean_correlation_and_boundary_contract(self):
        from analyze import correlation,boundary_mask
        self.assertTrue(np.isnan(correlation([0,1,2],[False,False,False])))
        assert_allclose(correlation([0,1,2],[False,True,True]),np.sqrt(3)/2)
        y=np.array([[0,0,255],[0,1,255],[0,1,1]]);valid=y!=255
        assert_allclose(boundary_mask(y,valid),valid)


if __name__=='__main__':unittest.main()
