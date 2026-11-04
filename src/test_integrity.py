import unittest
import numpy as np
from experiment import relation,corrected,metrics,review_policy
class IntegrityTests(unittest.TestCase):
 def test_zero_alpha_identity(self):
  p=np.array([[.2,.8]]);c=np.array([[0,.3],[.2,0]])
  np.testing.assert_array_equal(corrected(p,c,np.array([.5,.5]),0),p)
 def test_relation_no_self_edges(self):
  c,prior=relation(np.array([[1,1],[1,0],[0,1]]));np.testing.assert_array_equal(np.diag(c),[0,0]);self.assertTrue(np.isfinite(c).all())
 def test_missing_train_label_finite(self):
  c,_=relation(np.array([[1,0],[0,0]]));self.assertTrue(np.isfinite(c).all())
 def test_perfect_predictions(self):
  from experiment import LABELS
  y=np.eye(len(LABELS),dtype=int);self.assertEqual(metrics(y,y,.5)['macro_f1'],1)
 def test_review_policy_never_claims_unmet_target(self):
  y=np.array([[1,0],[0,1]]);p=1-y
  self.assertFalse(review_policy(y,p,np.array([.5,.5]))['target_met'])
if __name__=='__main__':unittest.main()
