import unittest,numpy as np
from audit_connected_material_volume_157 import audit
class VolumeAuditTests(unittest.TestCase):
 def setUp(self):
  self.P=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-1.],[0.,0.,2.]])
  self.T=np.array([[0,1,2,3],[0,2,1,4]]);self.M=np.array([1,2])
 def test_shared_interface_and_closed_union(self):
  r,a=audit(self.P,self.T,self.M)
  self.assertTrue(r['static_topology_and_positivity_pass']);self.assertEqual(r['shared_skin_fat_interface_triangles'],1)
  self.assertEqual(r['components']['union']['count'],1);self.assertEqual(r['boundary_triangles'],6)
  self.assertAlmostEqual(r['material_reference_volume_m3']['1'],1/6)
 def test_negative_cell_is_failure(self):
  T=self.T.copy();T[1,[1,2]]=T[1,[2,1]]
  r,a=audit(self.P,T,self.M)
  self.assertEqual(r['nonpositive_or_invalid_cells'],1);self.assertFalse(r['static_topology_and_positivity_pass'])
  self.assertEqual(r['orientation_mismatched_internal_faces'],1)
 def test_duplicate_tetrahedron_rejected(self):
  with self.assertRaisesRegex(ValueError,'duplicate'):audit(self.P,np.repeat(self.T[:1],2,axis=0),self.M)
 def test_nonmanifold_face_is_failure(self):
  r,a=audit(self.P,np.vstack([self.T,[0,1,2,5]]),[1,2,1])
  self.assertEqual(r['nonmanifold_faces'],1);self.assertFalse(r['static_topology_and_positivity_pass'])
 def test_nonfinite_or_materialless_rejected(self):
  P=self.P.copy();P[0,0]=np.nan
  with self.assertRaisesRegex(ValueError,'invalid'):audit(P,self.T,self.M)
  with self.assertRaisesRegex(ValueError,'material'):audit(self.P,self.T,[1,3])
if __name__=='__main__':unittest.main(verbosity=2)
