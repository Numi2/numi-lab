import unittest,importlib.util,numpy as np
from pathlib import Path
p=Path(__file__).with_name('ray_envelope_016.py');s=importlib.util.spec_from_file_location('ray016',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class RayTest(unittest.TestCase):
 def test_finite_triangle_exit_and_winding(self):
  t=np.array([[[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]]])
  np.testing.assert_allclose(m.ray_outward_exits(np.array([.2,.2,0.]),np.array([0.,0.,1.]),t,2.),[1.])
  self.assertEqual(len(m.ray_outward_exits(np.array([2.,2.,0.]),np.array([0.,0.,1.]),t,2.)),0)
  self.assertEqual(len(m.ray_outward_exits(np.array([.2,.2,0.]),np.array([0.,0.,1.]),t[:,[0,2,1]],2.)),0)
 def test_cap_and_parallel(self):
  t=np.array([[[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]]])
  self.assertEqual(len(m.ray_outward_exits(np.array([.2,.2,0.]),np.array([0.,0.,1.]),t,.5)),0)
  self.assertEqual(len(m.ray_outward_exits(np.array([.2,.2,0.]),np.array([1.,0.,0.]),t,2.)),0)
 def test_empty_input(self):
  self.assertEqual(len(m.ray_outward_exits(np.zeros(3),np.array([0.,0.,1.]),np.zeros((0,3,3)),1.)),0)
if __name__=='__main__':unittest.main()

