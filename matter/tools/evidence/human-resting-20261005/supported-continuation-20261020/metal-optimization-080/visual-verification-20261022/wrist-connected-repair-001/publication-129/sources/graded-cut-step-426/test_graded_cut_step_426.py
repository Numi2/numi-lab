import importlib.util
from pathlib import Path
import unittest,numpy as np
p=Path(__file__).with_name("graded_cut_step_426.py")
sp=importlib.util.spec_from_file_location("cuts426",p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
class ScaleTests(unittest.TestCase):
 def solve(self,rows,upper=None,budgets=None):
  return m.solve_scale_field(np.array([[0,1,2]]),np.ones(4)if upper is None else upper,np.array([3]),np.array([[0,1],[1,2]]),np.ones(2)if budgets is None else budgets,np.array([[1.,0,0]]*4),rows)
 def test_restricts_witness_and_keeps_other_motion(self):
  x,r=self.solve([dict(nodes=np.array([0]),coefficients=np.array([[-1.,0,0]]),lower=-.25)])
  np.testing.assert_allclose(x,[.25,1,1,0]);self.assertLessEqual(r["normalized_max_row_violation"],1e-8)
 def test_gradient_constraint_spreads_cap(self):
  x,_=self.solve([dict(nodes=np.array([0]),coefficients=np.array([[-1.,0,0]]),lower=-.25)],budgets=np.array([.1,.1]))
  np.testing.assert_allclose(x,[.25,.35,.45,0])
 def test_never_increases_existing_scales(self):
  u=np.array([.2,.3,.4,0]);x,_=self.solve([],upper=u);np.testing.assert_array_less(x,u+1e-12)
 def test_impossible_row_rejected(self):
  with self.assertRaises(ValueError):self.solve([dict(nodes=np.array([0]),coefficients=np.array([[0.,0,0]]),lower=1.)])
 def test_all_zero_feasible(self):
  x,_=self.solve([dict(nodes=np.array([0]),coefficients=np.array([[-1.,0,0]]),lower=0.)],budgets=np.array([.1,.1]));np.testing.assert_allclose(x,[0,.1,.2,0],atol=1e-12)
if __name__=="__main__":unittest.main()
