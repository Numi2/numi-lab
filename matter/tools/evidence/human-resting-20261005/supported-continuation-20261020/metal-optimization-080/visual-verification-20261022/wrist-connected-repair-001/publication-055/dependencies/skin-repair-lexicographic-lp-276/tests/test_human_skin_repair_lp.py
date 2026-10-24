import importlib.util
from pathlib import Path
import unittest
import numpy as np
path=Path(__file__).resolve().parents[1]/'tools/human_skin_repair_lp.py'
if not path.exists(): path=Path('/tmp/human_skin_repair_lp_276.py')
spec=importlib.util.spec_from_file_location('repair_lp276',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class LexicographicRepairTests(unittest.TestCase):
 def simple(self,scales=np.ones(3)):
  # x,u,s: contact x+s>=.5; u>=|x|.
  A=np.array([[-1,0,-1],[1,-1,0],[-1,-1,0]],dtype=float)*scales[:,None]
  return m.solve_contact_lp_lexicographic([0,1,1e9],A,np.array([-.5,0,0])*scales,[(-1,1),(0,None),(0,None)],2,1)
 def test_feasibility_before_shape(self):
  r,info=self.simple();self.assertTrue(r.success,r.message)
  self.assertAlmostEqual(r.x[0],.5,places=6);self.assertLess(info['contact_slack_sum'],2e-7)
  self.assertTrue(info['phase_two_run']);self.assertLess(info['maximum_original_constraint_violation'],1e-8)
 def test_row_units_do_not_change_solution(self):
  reference,_=self.simple();r,_=self.simple(np.array([1e12,1e-12,1e6]));self.assertTrue(r.success,r.message)
  np.testing.assert_allclose(r.x,reference.x,atol=1e-7)
 def test_conflicting_contacts_remain_explicit(self):
  # x+s0>=1 and -x+s1>=1, with u>=|x|.
  A=[[-1,0,-1,0],[1,0,0,-1],[1,-1,0,0],[-1,-1,0,0]]
  r,info=m.solve_contact_lp_lexicographic([0,1,1e12,1e12],A,[-1,-1,0,0],[(-2,2),(0,None),(0,None),(0,None)],2,2)
  self.assertTrue(r.success);self.assertAlmostEqual(info['minimum_contact_slack_sum'],2.)
  self.assertAlmostEqual(r.x[0],0.,places=6);self.assertGreater(info['contact_slack_max'],.999)
 def test_hard_geometry_infeasibility_not_hidden(self):
  r,info=m.solve_contact_lp_lexicographic([0,1],[[0,0]],[-1],[(-1,1),(0,None)],1,1)
  self.assertFalse(r.success);self.assertFalse(info['phase_two_run']);self.assertEqual(r.status,2)
 def test_bad_domains_and_nonfinite_input_rejected(self):
  args=([0,1],[[1,1]],[0],[(-1,1),(0,None)],1,1)
  for kwargs in ({'slack_count':0},{'slack_start':-1},{'slack_start':True},{'slack_sum_tolerance':-1}):
   vals=list(args);extra={}
   for k,v in kwargs.items():
    if k=='slack_count':vals[5]=v
    elif k=='slack_start':vals[4]=v
    else:extra[k]=v
   with self.assertRaises(ValueError):m.solve_contact_lp_lexicographic(*vals,**extra)
  for A,b,bounds in (([[np.nan,0]],[0],args[3]),([[1,0]],[np.inf],args[3]),([[1,0]],[0],[(-1,1),(None,None)])):
   with self.assertRaises(ValueError):m.solve_contact_lp_lexicographic(args[0],A,b,bounds,1,1)
if __name__=='__main__':unittest.main()
