import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
from scipy import sparse
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D")
OWNER=D/"surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
HELPER=D/"D/soft-target-nonregression-470/soft_target_nonregression_470.py"
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
class Solver470Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.owner=load("owner408_470_unit",OWNER); cls.helper=load("solver470_unit",HELPER)
 def args(self):
  return (sparse.csr_matrix([[-1.,0.]]),np.array([-.25]),np.zeros(2),np.array([-1.,-1.]),np.array([.25,1.]),
   sparse.csr_matrix([[1.,0.],[-1.,0.]]),np.array([1.,1.]),np.array([.75,.25]),sparse.eye(2,format="csr"),5.)
 def test_opposing_soft_rows_trade_off_and_hard_bound_remains(self):
  args=self.args(); base=self.owner.solve_sparse_l1_clearance_in_basis(*args); result=self.helper.make_solver(self.owner)(*args)
  self.assertTrue(base["success"],base); self.assertTrue(result["success"],result)
  self.assertLessEqual(abs(float(base["delta"][0])),2e-7)
  self.assertGreater(float(result["delta"][0]),.24); self.assertLessEqual(float(result["delta"][0]),.2500002)
  residual=args[0]@result["delta"]-args[1]; self.assertGreaterEqual(float(residual[0]),-2e-7)
  self.assertEqual(result["negative_soft_target_progress_rows"],1)
  self.assertEqual(result["no_regression_rows_enforced"],0); self.assertTrue(result["soft_target_regression_allowed"])
  self.assertTrue(result["stage1_output_certificate"]["passed"])
  self.assertEqual(result["stage1_output_certificate"]["checked_no_regression_rows"],0)
 def test_empty_targets_and_bad_dimensions(self):
  A=sparse.csr_matrix((0,2)); args=(A,np.zeros(0),np.zeros(2),np.full(2,-1.),np.ones(2),sparse.csr_matrix((0,2)),np.zeros(0),np.zeros(0),sparse.eye(2,format="csr"),2.)
  result=self.helper.make_solver(self.owner)(*args); self.assertTrue(result["success"],result); self.assertEqual(result["negative_soft_target_progress_rows"],0)
  with self.assertRaisesRegex(ValueError,"dimensions"):
   self.helper.make_solver(self.owner)(A,np.zeros(0),np.zeros(2),np.full(2,-1.),np.ones(2),sparse.csr_matrix((1,2)),np.zeros(0),np.zeros(0),sparse.eye(2,format="csr"),2.)
 def test_nonpositive_h_and_negative_weights_rejected(self):
  args=list(self.args()); args[6]=np.array([0.,1.])
  with self.assertRaisesRegex(ValueError,"invalid output bounds"): self.helper.make_solver(self.owner)(*args)
  args=list(self.args()); args[7]=np.array([1.,-.1])
  with self.assertRaisesRegex(ValueError,"invalid output bounds"): self.helper.make_solver(self.owner)(*args)
if __name__=="__main__": unittest.main()
