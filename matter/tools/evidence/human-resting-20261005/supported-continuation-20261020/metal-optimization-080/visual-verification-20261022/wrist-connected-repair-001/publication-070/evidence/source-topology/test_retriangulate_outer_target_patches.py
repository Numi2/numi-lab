import unittest
import numpy as np
import retriangulate_outer_target_patches as m

class RetriangulationTests(unittest.TestCase):
 def test_hexagon_has_fourteen_unique_triangulations(self):
  cycle=(10,11,12,13,14,15)
  tris=m.triangulations(cycle)
  self.assertEqual(len(tris),14)
  self.assertEqual(len({tuple(sorted(tuple(sorted(t)) for t in x)) for x in tris}),14)
 def test_all_hexagon_triangulations_preserve_directed_boundary(self):
  cycle=(0,1,2,3,4,5)
  source=np.asarray([(cycle[0],cycle[i],cycle[i+1]) for i in range(1,5)],dtype=np.int64)
  boundary=m.directed_boundary(source)[1]
  for tris in m.triangulations(cycle):
   self.assertEqual(m.directed_boundary_set(np.asarray(tris,dtype=np.int64)),boundary)
 def test_point_triangle_distance_inside_and_outside(self):
  tri=np.asarray([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
  q=np.asarray([[.2,.2,.3],[1.2,0.,0.]])
  d=m.point_triangle_distances(q,tri)
  self.assertAlmostEqual(float(d[0,0]),.3,places=12)
  self.assertAlmostEqual(float(d[1,0]),.2,places=12)

if __name__=="__main__":unittest.main()
