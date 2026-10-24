import unittest,importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location("scope_geometry",Path(__file__).resolve().parents[1]/"tools/human_skin_patch_geometry.py")
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
class ConnectedPlaneRegionTests(unittest.TestCase):
 def run_scope(self,x,f,seed=0,c=0):
  return g.connected_positive_plane_face_component(np.asarray(x,float),np.asarray(f),axis=2,coordinate=c,seed_face=seed)
 def test_connected_original_face_ids_are_retained(self):
  x=[[0,0,1],[1,0,1],[0,1,1],[1,1,1],[4,0,1],[5,0,1],[4,1,1]]
  f=[[0,1,2],[1,3,2],[4,5,6]]
  np.testing.assert_array_equal(self.run_scope(x,f),[0,1])
  np.testing.assert_array_equal(self.run_scope(x,f,seed=2),[2])
 def test_edge_removed_by_clipping_does_not_join_regions(self):
  x=[[0,0,-1],[1,0,-1],[0,1,1],[0,-1,1]]
  np.testing.assert_array_equal(self.run_scope(x,[[0,1,2],[1,0,3]]),[0])
 def test_edge_on_plane_does_not_join_regions(self):
  x=[[0,0,0],[1,0,0],[0,1,1],[0,-1,1]]
  np.testing.assert_array_equal(self.run_scope(x,[[0,1,2],[1,0,3]]),[0])
 def test_crossing_edge_survives_clipping(self):
  x=[[0,0,-1],[1,0,1],[0,1,1],[0,-1,1]]
  np.testing.assert_array_equal(self.run_scope(x,[[0,1,2],[1,0,3]]),[0,1])
 def test_coordinate_seams_not_merged(self):
  x=[[0,0,1],[1,0,1],[0,1,1],[0,0,1],[1,0,1],[0,-1,1]]
  np.testing.assert_array_equal(self.run_scope(x,[[0,1,2],[4,3,5]]),[0])
 def test_nonmanifold_active_edge_fails(self):
  x=[[0,0,1],[1,0,1],[0,1,1],[0,-1,1],[.5,.5,2]]
  with self.assertRaisesRegex(ValueError,"nonmanifold"):self.run_scope(x,[[0,1,2],[1,0,3],[0,1,4]])
 def test_inactive_nonmanifold_edge_does_not_join(self):
  x=[[0,0,-1],[1,0,-1],[0,1,1],[0,-1,1],[.5,.5,2]]
  np.testing.assert_array_equal(self.run_scope(x,[[0,1,2],[1,0,3],[0,1,4]]),[0])
 def test_axis_uses_selected_coordinate(self):
  x=np.array([[1,0,0],[1,1,0],[1,0,1]])
  np.testing.assert_array_equal(g.connected_positive_plane_face_component(x,[[0,1,2]],axis=0,coordinate=0,seed_face=0),[0])
 def test_invalid_inputs_fail_before_casting(self):
  x=np.array([[0,0,1],[1,0,1],[0,1,1]])
  cases=[(x,np.array([[0.,1.,2.]]),2,0,0),(x,[[0,1,3]],2,0,0),(x,[[-1,1,2]],2,0,0),(x,[[0,0,2]],2,0,0),(x,[[0,1,2]],True,0,0),(x,[[0,1,2]],2,np.nan,0),(x,[[0,1,2]],2,[0],0),(x,[[0,1,2]],2,0,True),(x,[[0,1,2]],2,0,9),(x,[[0,1,2]],2,2,0)]
  for xx,ff,a,c,s in cases:
   with self.subTest(a=a,c=c,seed=s),self.assertRaises(ValueError):
    g.connected_positive_plane_face_component(xx,ff,axis=a,coordinate=c,seed_face=s)
if __name__=="__main__":unittest.main()
