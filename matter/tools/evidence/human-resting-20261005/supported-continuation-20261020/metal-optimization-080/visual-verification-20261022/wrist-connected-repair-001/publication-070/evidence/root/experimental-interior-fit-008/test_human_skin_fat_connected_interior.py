import unittest
import numpy as np
from human_skin_fat_connected_repair import fit_connected_layer_interior
class ConnectedInterior(unittest.TestCase):
 def fixture(self):
  R=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]],np.float32)
  return R,np.array([[0,1,2,3]],np.int64)
 def test_connected_untangling_anchors_source_and_retains_reference(self):
  R,T=self.fixture();X=R.copy();X[3,2]=-.001;before=R.copy()
  Y,r=fit_connected_layer_interior(X,R,T,np.array([1,2,3]),max_iterations=32)
  self.assertTrue(r['local_cell_gates_pass'],r)
  np.testing.assert_array_equal(Y[0].view(np.uint32),X[0].view(np.uint32))
  np.testing.assert_array_equal(R,before)
  self.assertFalse(r['geometry_admitted']);self.assertEqual(r['accepted_physical_steps'],0)
  for row in r['history']:self.assertLessEqual(row['merit'],r['starting']['merit'])
 def test_untangling_cannot_fix_an_anchored_bad_cell(self):
  R,T=self.fixture();X=np.vstack([R,[[.1,.1,.1]]]).astype(np.float32);X[3,2]=-.001
  ref=np.vstack([R,[[.1,.1,.1]]]).astype(np.float32)
  Y,r=fit_connected_layer_interior(X,ref,T,np.array([4]),max_iterations=4)
  self.assertFalse(r['local_cell_gates_pass']);self.assertEqual(r['final']['invalid_cells'],1)
  np.testing.assert_array_equal(Y.view(np.uint32),X.view(np.uint32))
 def test_invalid_domains_fail_closed(self):
  R,T=self.fixture()
  for free in [np.array([],dtype=int),np.array([1.,2.]),np.array([1,1]),np.array([9])]:
   with self.assertRaises(ValueError):fit_connected_layer_interior(R,R,T,free)
  with self.assertRaises(ValueError):fit_connected_layer_interior(R.astype(float),R,T,np.array([1]))
  with self.assertRaises(ValueError):fit_connected_layer_interior(R,R,T,np.array([1]),max_iterations=True)
if __name__=='__main__':unittest.main()
