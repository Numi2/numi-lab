import unittest,numpy as np
from human_skin_p1_stream_clip import TetBroadphase,free_capacity
class FullDomainTests(unittest.TestCase):
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


import unittest,numpy as np
from human_skin_p1_stream_clip import CapacityBroadphase
class CapacityDomainTests(unittest.TestCase):
 def test_unsafe_convex_hull_contains_all_sampled_low_capacity_points(self):
  rng=np.random.default_rng(25);x=rng.uniform(-1,1,(120,3)).astype(np.float32);t=np.arange(120).reshape(-1,4)
  fixed=rng.random(120)<.45;b=CapacityBroadphase(x,t,fixed,.341)
  weights=rng.exponential(1,(5000,4));weights/=weights.sum(1)[:,None]
  for ti in range(len(t)):
   cap=weights[:,~fixed[t[ti]]].sum(1);p=weights[cap<.341]@x[t[ti]].astype(float)
   if len(p):
    self.assertTrue(np.all(p>=b.lo[ti]-1e-14));self.assertTrue(np.all(p<=b.hi[ti]+1e-14))
    # Point triangles exercise the complete sphere/bin/AABB path.
    got=b.candidates(np.repeat(p[:10,None,:],3,axis=1))
    self.assertTrue(all(ti in rows for rows in got))
 def test_free_only_excluded_fixed_only_complete(self):
  x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[2,0,0],[3,0,0],[2,1,0],[2,0,1]],np.float32)
  f=np.array([0,0,0,0,1,1,1,1],bool);b=CapacityBroadphase(x,[[0,1,2,3],[4,5,6,7]],f,.341)
  np.testing.assert_array_equal(b.eligible,[1]);np.testing.assert_array_equal(b.lo[1],np.nextafter([2.,0.,0.],-np.inf))
  with self.assertRaises(ValueError):CapacityBroadphase(x,[[0,1,2,3]],f,0)
if __name__=='__main__':unittest.main()
