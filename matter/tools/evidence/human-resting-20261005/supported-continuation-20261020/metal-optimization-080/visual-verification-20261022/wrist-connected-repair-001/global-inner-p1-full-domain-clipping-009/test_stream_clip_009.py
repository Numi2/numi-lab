import unittest,numpy as np
from stream_clip_009 import TetBroadphase,free_capacity
class Tests(unittest.TestCase):
 def test_enclosure_matches_all_aabb_candidates(self):
  rng=np.random.default_rng(71)
  x=rng.uniform(-.1,.1,(800,3)).astype(np.float32);t=np.arange(800).reshape(-1,4)
  p=rng.uniform(-.2,.2,(300,3,3))
  p[0]=[[-100,0,0],[100,0,0],[0,1e-8,0]] # long skinny triangle
  b=TetBroadphase(x,t)
  for tri,got in zip(p,b.candidates(p)):
   want=np.flatnonzero(np.all(b.hi>=tri.min(0)-2e-12,axis=1)&np.all(b.lo<=tri.max(0)+2e-12,axis=1))
   np.testing.assert_array_equal(got,want)
 def test_empty_and_shared_boundary(self):
  x=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);b=TetBroadphase(x,[[0,1,2,3]])
  self.assertEqual(b.candidates(np.zeros((0,3,3))),[])
  np.testing.assert_array_equal(b.candidates(np.asarray([[[0,0,0],[0,1,0],[1,0,0]]]))[0],[0])
 def test_fixed_zero_capacity_and_domain_validation(self):
  f=np.asarray([1,1,1,0],bool)
  np.testing.assert_allclose(free_capacity([[.2,.3,.4,.1]],[0,1,2,3],f),[.1])
  with self.assertRaises(ValueError):free_capacity([[-.1,.3,.4,.4]],[0,1,2,3],f)
  with self.assertRaises(ValueError):TetBroadphase(np.zeros((4,3)),[[0,1,2,4]])
if __name__=="__main__":unittest.main()
