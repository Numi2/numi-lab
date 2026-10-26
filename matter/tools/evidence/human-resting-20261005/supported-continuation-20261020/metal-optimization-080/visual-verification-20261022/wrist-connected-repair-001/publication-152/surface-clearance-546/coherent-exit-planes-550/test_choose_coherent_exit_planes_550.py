import unittest,numpy as np
from choose_coherent_exit_planes_550 import choose_plane_candidate,expand_selection_row550
class TestChooser(unittest.TestCase):
 def setUp(self):
  self.x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
  self.up=np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
  self.down=np.array([[0.,0.,-1.],[0.,1.,-1.],[1.,0.,-1.]])
 def c(self,tri,tid=3,ray=(0,0,1),matches=True):
  return dict(identity_matches=matches,triangle_stable_id=tid,source_kind='nearest_boundary',source_pair_index=0,triangle_world_mm=tri,certified_direction_world_unit=np.array(ray,float))
 def test_positive_exit_eligible(self):
  c=self.c(self.up);r=choose_plane_candidate(self.x,[c],c);self.assertEqual(r['mode'],'certified_endpoint_facet_minimum_corner_deficit')
 def test_opposed_exit_rejected(self):
  c=self.c(self.down);r=choose_plane_candidate(self.x,[c],c);self.assertIn('fallback',r['mode'])
 def test_identity_mismatch_rejected(self):
  c=self.c(self.up,matches=False);r=choose_plane_candidate(self.x,[c],c);self.assertIn('fallback',r['mode'])
 def test_stable_id_breaks_tie(self):
  a=self.c(self.up,7);b=self.c(self.up,3);r=choose_plane_candidate(self.x,[a,b],a);self.assertEqual(r['triangle_stable_id'],3)
 def test_minimum_deficit(self):
  a=self.c(self.up,3);b=self.c(self.up*.5,7);r=choose_plane_candidate(self.x,[a,b],a);self.assertEqual(r['triangle_stable_id'],7)
 def test_fallback_per_original_facet(self):
  g=dict(selection_mode='original_crossed_facet_fallback_no_eligible_endpoint_plane')
  a=expand_selection_row550(g,3,self.up,self.x,(0,0,1),0);b=expand_selection_row550(g,4,self.down,self.x,(0,0,-1),1)
  self.assertEqual(a['selected_plane_triangle_stable_id'],3);self.assertEqual(b['selected_plane_triangle_stable_id'],4);self.assertTrue(np.array_equal(b['selected_plane_world_vertices_mm'],self.down))
 def test_degenerate_rejected(self):
  a=self.c(np.zeros((3,3)))
  with self.assertRaises(ValueError):choose_plane_candidate(self.x,[a],a)
if __name__=='__main__':unittest.main()
