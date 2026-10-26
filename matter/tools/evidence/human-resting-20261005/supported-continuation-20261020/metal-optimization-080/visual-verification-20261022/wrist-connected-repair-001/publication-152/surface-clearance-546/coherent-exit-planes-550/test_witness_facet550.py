import unittest
import numpy as np
from coherent_exit_plane_adapter550 import witness_plane_rows550
class TestWitnessFacet(unittest.TestCase):
 def setUp(self):
  self.tri=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
  self.arr=dict(bank_body=np.array([1,1]),bank_stable_bone_id=np.array([2,2]),bank_triangle_stable_id=np.array([3,3]),original_crossed_triangles_world_mm=np.array([self.tri,self.tri]))
  self.p=np.array([[.0001,.0001,.00001]],np.float32)
  self.w=dict(node=0,body=1,bone=2,triangle=3,triangle_world_mm=self.tri.tolist())
 def run_rows(self):return witness_plane_rows550(dict(witnesses=[self.w]),self.p,self.arr)
 def test_baseline_feasible_and_plane_excludes_inside(self):
  rows,receipt=self.run_rows();r=rows[0]
  self.assertEqual(len(rows),1);self.assertLess(r['lower'],0.)
  n=r['coefficients'][0];inside=np.array([.1,.1,-.01])
  self.assertLess((inside-self.p[0].astype(float)*1000.)@n,r['lower'])
  self.assertGreater(receipt[0]['emission_margin_mm'],0.)
 def test_nearest_other_facet_cannot_substitute(self):
  self.w['triangle']=4
  with self.assertRaisesRegex(ValueError,'absent'):self.run_rows()
 def test_geometry_mismatch_rejected(self):
  self.w['triangle_world_mm'][0][2]=.1
  with self.assertRaisesRegex(ValueError,'mismatch'):self.run_rows()
 def test_inside_baseline_rejected(self):
  self.p[0,2]=-1e-5
  with self.assertRaisesRegex(ValueError,'zero-step'):self.run_rows()
 def test_duplicate_and_bounds(self):
  with self.assertRaisesRegex(ValueError,'duplicate'):witness_plane_rows550(dict(witnesses=[self.w,self.w]),self.p,self.arr)
  self.w['node']=1
  with self.assertRaisesRegex(ValueError,'bounds'):self.run_rows()
 def test_conflicting_registered_geometry_rejected(self):
  self.arr['original_crossed_triangles_world_mm'][1,0,2]=.001
  with self.assertRaisesRegex(ValueError,'mismatch'):self.run_rows()
if __name__=='__main__':unittest.main()
