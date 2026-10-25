import unittest,importlib.util,numpy as np
from pathlib import Path
s=importlib.util.spec_from_file_location('g',Path(__file__).with_name('global_step_backtracking_275.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class GlobalTests(unittest.TestCase):
 def test_uniform_free_scaling_and_anchor_cutoff(self):
  a=np.array([0.,1.,1.,1.]);F=np.array([[0,1,2],[1,3,2]])
  b,r=m.reduce_global_step_scales(F,a,[0],[],anchors=[0],zero_below=1/16)
  np.testing.assert_array_equal(b,[0,.5,.5,.5]);np.testing.assert_array_equal(a,[0,1,1,1]);self.assertFalse(r['geometry_certified'])
  b,r=m.reduce_global_step_scales(F,np.array([0.,1/16,1/16,1/16]),[0],[],anchors=[0],zero_below=1/16);np.testing.assert_array_equal(b,np.zeros(4))
 def test_no_witness_and_bad_domain_rejected(self):
  with self.assertRaises(ValueError):m.reduce_global_step_scales(np.array([[0,1,2]]),np.ones(3),[],[],anchors=[],zero_below=1/16)
  with self.assertRaises(ValueError):m.reduce_global_step_scales(np.array([[0,1,4]]),np.ones(3),[0],[],anchors=[],zero_below=1/16)
if __name__=='__main__':unittest.main()
