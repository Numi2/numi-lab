import unittest,numpy as np
from component_area_backtracking549 import components,line_search
class TestComponents(unittest.TestCase):
 def test_disjoint_faces(self):
  g,p=components(6,np.array([[0,1,2],[3,4,5]]),np.arange(6),np.empty((0,2),int));self.assertEqual(p['sizes'],[3,3])
 def test_pair_merges_complete_stencils(self):
  g,p=components(6,np.array([[0,1,2],[3,4,5]]),np.arange(6),np.array([[0,1]]));self.assertEqual(p['sizes'],[6])
 def test_shared_fixed_node_does_not_couple(self):
  g,p=components(5,np.array([[0,1,2],[0,3,4]]),np.array([1,2,3,4]),np.empty((0,2),int));self.assertEqual(p['sizes'],[2,2])
 def test_zero_linear_derivative_still_coupled(self):
  g,p=components(3,np.array([[0,1,2]]),np.array([0,2]),np.empty((0,2),int));self.assertEqual(p['sizes'],[2])
 def test_invalid_pair(self):
  with self.assertRaises(ValueError):components(3,np.array([[0,1,2]]),np.arange(3),np.array([[0,1]]))
 def test_duplicate_free(self):
  with self.assertRaises(ValueError):components(3,np.array([[0,1,2]]),np.array([1,1]),np.empty((0,2),int))
 def test_independent_alpha_and_fixed_bits(self):
  b=np.zeros((3,3),np.float32);t=b.copy();t[:2,0]=[1,1]
  def gate(x):return bool(x[0,0]<=.2 and x[1,0]<=1),{}
  x,r=line_search(b,t,np.array([0,1]),[np.array([0]),np.array([1])],gate)
  self.assertEqual([i['selected_alpha'] for i in r],[.125,1]);self.assertTrue(np.array_equal(x[2].view(np.uint32),b[2].view(np.uint32)))
 def test_failed_region_keeps_other_progress(self):
  b=np.zeros((2,3),np.float32);t=b.copy();t[:,0]=1
  x,r=line_search(b,t,np.arange(2),[np.array([0]),np.array([1])],lambda x:(bool(x[0,0]==0),{}),max_halvings=2)
  self.assertEqual(r[0]['selected_alpha'],None);self.assertEqual(x[1,0],1)
 def test_bad_baseline_rejected(self):
  b=np.zeros((1,3),np.float32)
  with self.assertRaises(ValueError):line_search(b,b,np.array([0]),[np.array([0])],lambda x:(False,{}))
 def test_malformed_partition_rejected(self):
  b=np.zeros((2,3),np.float32)
  with self.assertRaises(ValueError):line_search(b,b,np.arange(2),[np.array([0])],lambda x:(True,{}))
if __name__=='__main__':unittest.main()
