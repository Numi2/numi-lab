
import unittest
import numpy as np
from compare_matched_deformation_339 import retained_node_map, compare_matched
def fixture(duplicate=False):
 p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],float)
 if duplicate:p=np.vstack([p,p[:1]])
 n=len(p);t=np.array([[0,1,2,3],[1,2,3,4]]+([[5,1,2,3]] if duplicate else []),np.int32)
 return dict(objects={"skin-fat":0},rest=p,u=p@np.diag([.001,.002,.003]),mass=np.ones(n),
             tag=np.zeros(n),region=np.full(n,4,np.int32),obj=np.zeros(n,np.int32),
             tets=t,treg=np.full(len(t),4,np.int32),tobj=np.zeros(len(t),np.int32))
def clone(a):return {k:(v.copy() if isinstance(v,np.ndarray) else v.copy()) for k,v in a.items()}
class Mapping(unittest.TestCase):
 def test_identity(self):
  a=fixture();m,r=retained_node_map(a,clone(a));np.testing.assert_array_equal(m,np.arange(5));self.assertEqual(r["duplicate_nodes"],0)
 def test_reordered_nodes(self):
  a=fixture();b=clone(a);order=np.array([4,0,3,1,2]);inv=np.argsort(order)
  for k in ["rest","u","mass","tag","region","obj"]:b[k]=b[k][order]
  b["tets"]=inv[b["tets"]]
  m,_=retained_node_map(a,b);np.testing.assert_array_equal(m,order)
 def test_coincident_nodes_unchanged(self):
  a=fixture(True);m,r=retained_node_map(a,clone(a));np.testing.assert_array_equal(m,np.arange(6));self.assertEqual(r["duplicate_nodes"],2)
 def test_coincident_changed_topology_rejected(self):
  a=fixture(True);b=clone(a);b["tets"][0]=[5,1,2,3]
  with self.assertRaisesRegex(ValueError,"ambiguous"):retained_node_map(a,b)
 def test_wrong_owner_rejected(self):
  a=fixture();b=clone(a);b["objects"]={"other":0}
  with self.assertRaisesRegex(ValueError,"identity"):retained_node_map(a,b)
 def test_wrong_region_rejected(self):
  a=fixture();b=clone(a);b["region"][0]=7
  with self.assertRaisesRegex(ValueError,"missing"):retained_node_map(a,b)
 def test_missing_coordinate_rejected(self):
  a=fixture();b=clone(a);b["rest"][0,0]=.25
  with self.assertRaisesRegex(ValueError,"missing"):retained_node_map(a,b)
 def test_changed_constraint_rejected(self):
  a=fixture();b=clone(a);b["tag"][0]=2
  with self.assertRaisesRegex(ValueError,"constraint"):retained_node_map(a,b)
 def test_duplicate_mass_change_rejected(self):
  a=fixture(True);b=clone(a);b["mass"][0]*=2
  with self.assertRaisesRegex(ValueError,"ambiguous"):retained_node_map(a,b)
 def test_affine_restriction(self):
  a=fixture();b=clone(a);b["u"]*=1.1
  r=compare_matched(a,b)
  self.assertEqual(r["discretization_at_reference_displacement"]["summary"]["maximum_relative_change"],0)
  self.assertGreater(r["candidate_actual_vs_restricted_reference_displacement"]["summary"]["maximum_relative_change"],.05)
  self.assertEqual(r["identical_cells"]["restricted_field_max_absolute_green_strain_difference"],0)
if __name__=="__main__":unittest.main()
