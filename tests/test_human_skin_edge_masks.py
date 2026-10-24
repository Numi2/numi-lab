import importlib.util
from pathlib import Path
import unittest
import itertools
import numpy as np
spec=importlib.util.spec_from_file_location('geometry254',Path(__file__).resolve().parents[1]/'tools/human_skin_patch_geometry.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
class EdgeMaskRegression(unittest.TestCase):
 def test_all_masks_windings_parameters_and_oblique_scales(self):
  fixture=np.array([[-.24387451,.16671881,.78889013],[-.24300053,.1654294,.79357094],[-.24234738,.16761126,.79395885]])
  for scale,reverse,mask,t in itertools.product((.01,1.,100.),(False,True),range(1,8),(.1,.5,.9)):
   x=fixture*scale;f=np.array([[0,2,1] if reverse else [0,1,2]])
   edges=[tuple(map(int,e)) for e in zip(f[0],np.roll(f[0],-1))]
   cuts={edge:t for k,edge in enumerate(edges) if mask&(1<<k)}
   y,h,p,e,w=g.split_surface_edges(x,f,cuts)
   np.testing.assert_array_equal(y[:3],x)
   np.testing.assert_allclose(y[3:],np.sum(x[e]*w[:,:,None],axis=1),rtol=0,atol=0)
   self.assertEqual(len(h),mask.bit_count()+1)
   np.testing.assert_array_equal(p,np.zeros(len(h),dtype=int))
   parent=np.cross(x[f[0,1]]-x[f[0,0]],x[f[0,2]]-x[f[0,0]])
   z=y[h];child=np.cross(z[:,1]-z[:,0],z[:,2]-z[:,0])
   self.assertTrue(np.all(child@parent>0))
   np.testing.assert_allclose(child.sum(axis=0),parent,rtol=2e-11,atol=1e-22)
 def test_adjacent_faces_keep_opposed_shared_edge_segments(self):
  x=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
  y,f,p,e,w=g.split_surface_edges(x,np.array([[0,1,2],[0,2,3]]),{(2,0):.3})
  edges={}
  for tri in f:
   for a,b in zip(tri,np.roll(tri,-1)):
    edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append((int(a),int(b)))
  for key in ((0,4),(2,4)):
   self.assertEqual(len(edges[key]),2);self.assertEqual(edges[key][0],edges[key][1][::-1])
if __name__=='__main__':unittest.main()
