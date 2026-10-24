import unittest
import numpy as np
from human_skin_patch_geometry import undirected_boundary_loops as loops
class BoundaryLoopTests(unittest.TestCase):
 def test_orientation_order_independent(self):
  e=np.array([[10,2],[7,10],[2,7]])
  for v in [e,e[::-1],e[:,::-1]]:
   self.assertEqual([x.tolist() for x in loops(v)],[[2,7,10]])
 def test_disconnected_cycles(self):
  e=np.array([[1,2],[2,3],[3,1],[10,11],[11,12],[12,10]])
  self.assertEqual([x.tolist() for x in loops(e)],[[1,2,3],[10,11,12]])
 def test_empty(self):
  self.assertEqual(loops(np.empty((0,2),np.int64)),[])
 def test_open_or_branched_rejected_before_walk(self):
  for e in [[[1,2],[2,3]],[[0,1],[1,2],[2,0],[0,3],[3,4],[4,0]]]:
   with self.assertRaisesRegex(ValueError,'degree two'):loops(np.array(e))
 def test_duplicate_self_invalid(self):
  for e in [[[1,2],[2,1]],[[1,1]],[[1,-1]],[[1.,2.]],[[1,2,3]]]:
   with self.assertRaises(ValueError):loops(np.array(e))
 def test_long_cycle_finishes(self):
  n=10000;e=np.c_[np.arange(n),np.roll(np.arange(n),-1)]
  out=loops(e);self.assertEqual(len(out),1);np.testing.assert_array_equal(out[0],np.arange(n))
if __name__=='__main__':unittest.main()
