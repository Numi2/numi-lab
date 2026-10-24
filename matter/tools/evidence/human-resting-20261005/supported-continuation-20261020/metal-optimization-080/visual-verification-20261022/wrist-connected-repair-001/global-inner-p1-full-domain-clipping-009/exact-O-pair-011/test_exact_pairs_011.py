import unittest
import numpy as np
from audit_exact_pairs_011 import ci,classify
def tri(p):return tuple(ci.float32_point_lattice_key(x) for x in np.asarray(p,np.float32))
class Tests(unittest.TestCase):
 def test_crossing_both_interiors(self):
  a=tri([[0,0,0],[1,0,0],[0,1,0]]);b=tri([[.25,.25,-1],[.25,.25,1],[.75,.25,0]])
  self.assertEqual(classify(a,b)[0],'transverse_both_interiors')
 def test_contact_is_not_classified_as_transverse_penetration(self):
  a=tri([[0,0,0],[1,0,0],[0,1,0]])
  self.assertEqual(classify(a,tri([[0,0,0],[1,0,0],[0,0,1]]))[0],'coincident_edge_contact')
  self.assertEqual(classify(a,tri([[0,0,0],[-1,0,1],[0,-1,1]]))[0],'point_contact')
  self.assertEqual(classify(a,a)[0],'coincident_triangle')
 def test_clear_and_coplanar_overlap(self):
  a=tri([[0,0,0],[1,0,0],[0,1,0]])
  self.assertEqual(classify(a,tri([[0,0,2],[1,0,2],[0,1,2]]))[0],'clear')
  self.assertEqual(classify(a,tri([[.25,0,0],[1,0,0],[.25,.75,0]]))[0],'coplanar_area_overlap')
if __name__=='__main__':unittest.main()
