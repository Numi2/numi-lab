import unittest,importlib.util,sys
from pathlib import Path
import numpy as np
from coherent_exit_plane_adapter550 import keyed_selection,build_bundle
O=Path('/Users/numi/Work/numi-human-connected-surface-069/D/D/fixed-bone-outward-plane-458/fixed_bone_outward_plane_458.py')
s=importlib.util.spec_from_file_location('original458test550',O);o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
class TestBundle(unittest.TestCase):
 def setUp(self):
  self.x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
  self.f=np.array([[0,1,2]]);self.pairs=[dict(face_row=0,body=1,bone_id=2,triangle_id=i) for i in [3,4]]
  self.bone={(1,2,3):np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]]),(1,2,4):np.array([[0.,0.,-1.],[0.,1.,-1.],[1.,0.,-1.]])}
 def test_coherent_plane_identity_and_equal_mass(self):
  sel={(0,1,2,i):dict(triangle_id=3,triangle_mm=self.bone[(1,2,3)],eligible=True) for i in [3,4]}
  r=build_bundle(o.build_fixed_bone_outward_plane_rows,sel,self.x,self.f,np.arange(3),np.array([],int),self.pairs,self.bone)
  self.assertEqual(r['G'].shape[0],6);self.assertAlmostEqual(r['weights'].sum(),1.);self.assertEqual({m['plane_triangle_id'] for m in r['metadata']},{3});self.assertEqual({m['crossed_triangle_id'] for m in r['metadata']},{3,4})
 def test_fallback_parity(self):
  sel={(0,1,2,i):dict(triangle_id=i,triangle_mm=self.bone[(1,2,i)],eligible=False) for i in [3,4]}
  r=build_bundle(o.build_fixed_bone_outward_plane_rows,sel,self.x,self.f,np.arange(3),np.array([],int),self.pairs,self.bone)
  old=o.build_fixed_bone_outward_plane_rows(self.x,self.f,np.arange(3),np.array([],int),self.pairs,self.bone)
  self.assertTrue(np.array_equal(r['G'].toarray(),old['G'].toarray()));self.assertTrue(np.array_equal(r['lower'],old['lower']))
 def test_inconsistent_group_rejected(self):
  sel={(0,1,2,i):dict(triangle_id=i,triangle_mm=self.bone[(1,2,i)],eligible=True) for i in [3,4]}
  with self.assertRaises(ValueError):build_bundle(o.build_fixed_bone_outward_plane_rows,sel,self.x,self.f,np.arange(3),np.array([],int),self.pairs,self.bone)
 def test_missing_identity_rejected(self):
  with self.assertRaises(ValueError):build_bundle(o.build_fixed_bone_outward_plane_rows,{},self.x,self.f,np.arange(3),np.array([],int),self.pairs,self.bone)
 def test_anchored_positive_preserved(self):
  sel={(0,1,2,i):dict(triangle_id=3,triangle_mm=self.bone[(1,2,3)],eligible=True) for i in [3,4]}
  r=build_bundle(o.build_fixed_bone_outward_plane_rows,sel,self.x,self.f,np.arange(3),np.array([0]),self.pairs,self.bone)
  self.assertEqual(r['diagnostics']['anchored_positive_lower_count'],2)
 def test_keyed_bad_shapes_rejected(self):
  with self.assertRaises(ValueError):keyed_selection({})
if __name__=='__main__':unittest.main()
