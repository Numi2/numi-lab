import unittest, importlib.util,sys
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('active_layer_refinement_008.py')
s=importlib.util.spec_from_file_location('active008',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a);r=a.owner403()
class ActiveLayerTests(unittest.TestCase):
 def fixture(self):
  X=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[.25,.25,.25]],np.float32)
  T=np.asarray([r.orient_positive(X,(*f,4)) for f in ((1,2,3),(0,3,2),(0,1,3),(0,2,1))])
  return X,T,np.arange(4),r.oriented_boundary_faces(T)
 def test_boundary_edges_are_never_selected(self):
  X,T,O,S=self.fixture()
  self.assertEqual(a.select_internal_outer_free_edges(T,O,S,np.arange(4),len(X)),[(0,4),(1,4),(2,4),(3,4)])
  # Declaring the free center as boundary must remove its incident edges.
  allfacets=np.concatenate([T[:,[0,1,2]],T[:,[0,1,3]],T[:,[0,2,3]],T[:,[1,2,3]]])
  self.assertEqual(a.select_internal_outer_free_edges(T,O,allfacets,np.arange(4),len(X)),[])
 def test_refinement_recovers_capacity_without_moving_shell(self):
  X,T,O,S=self.fixture();original=X.copy()
  point=np.asarray([.3,.3,0])*.978+X[4].astype(float)*.022
  minimum=[]
  for level in range(5):
   q=X[T].astype(float);A=np.stack((q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]),axis=2)
   uv=np.linalg.solve(A,np.broadcast_to(point,(len(T),3))-q[:,0])
   B=np.column_stack((1-uv.sum(axis=1),uv));rows=np.flatnonzero(B.min(axis=1)>=-1e-12)
   self.assertTrue(len(rows))
   cap=np.asarray([sum(b[~np.isin(T[i],O)]) for i,b in zip(rows,B[rows])])
   minimum.append(float(cap.min()))
   if cap.min()>=.341:break
   edges=a.select_internal_outer_free_edges(T,O,S,rows[cap<.341],len(X))
   X,T,parents,endpoints=r.split_edges_globally(X,T,edges)
   self.assertTrue(np.array_equal(X[:len(original)],original))
   boundary=r.compare_oriented_boundary(S,r.oriented_boundary_faces(T))
   self.assertTrue(boundary['exact_unoriented_set']);self.assertEqual(boundary['oriented_mismatch_count'],0)
   self.assertTrue(np.all(r.tetra_metrics(X,T)[0]>0))
  self.assertGreaterEqual(minimum[-1],.341)
  self.assertGreater(minimum[-1],minimum[0]*10)
 def test_genuine_fixed_only_constraints_get_no_free_support(self):
  X,T,O,S=self.fixture();O=np.arange(5)
  self.assertEqual(a.select_internal_outer_free_edges(T,O,S,np.arange(4),len(X)),[])
 def test_stale_tetra_domain_is_rejected(self):
  X,T,O,S=self.fixture()
  with self.assertRaises(ValueError):a.select_internal_outer_free_edges(T,O,S,np.asarray([4]),len(X))
if __name__=='__main__':unittest.main()

