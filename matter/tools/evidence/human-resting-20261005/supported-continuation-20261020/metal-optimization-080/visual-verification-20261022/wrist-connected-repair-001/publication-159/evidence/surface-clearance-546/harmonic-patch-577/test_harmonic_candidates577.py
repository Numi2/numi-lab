import unittest,numpy as np
import harmonic_candidates577 as h
class Tests(unittest.TestCase):
 def fixture(self):
  from scipy.spatial import Delaunay
  t=np.arange(7)*2*np.pi/7
  p=np.vstack((np.column_stack((np.cos(t),np.sin(t),np.zeros(7))),[[0,0,1],[.3,.1,-1]]))
  f=Delaunay(p[:,:2]).simplices.copy()
  for j,t in enumerate(f):
   if h.s569.orient(p[t[0],:2],p[t[1],:2],p[t[2],:2])<0:f[j,1],f[j,2]=f[j,2],f[j,1]
  return p,f
 def test_chart_all_nodes_positive(self):
  p,f=self.fixture();r,o,b,v=h.p571._boundary_and_vertices(p,f,np.arange(len(f)),48)
  for w in(False,True):
   uv=h.harmonic_chart(p,o,b,v,w);self.assertEqual(uv.shape,(len(v),2));self.assertTrue(np.isfinite(uv).all())
 def test_alternatives_preserve_geometry_ids_boundary(self):
  p,f=self.fixture();before=p.tobytes();a=h.patch_candidates(p,p,f,np.arange(len(f)))
  self.assertTrue(a);b=h.p571._boundary_and_vertices(p,f,np.arange(len(f)),48)[2]
  for nf in a:
   self.assertEqual(nf.shape,f.shape);self.assertEqual(set(nf.flat),set(f.flat))
   self.assertEqual(h.p571._boundary_and_vertices(p,nf,np.arange(len(nf)),48)[2],b)
  self.assertEqual(before,p.tobytes())
 def test_broken_patch_rejected(self):
  p,f=self.fixture();f[0]=f[1]
  with self.assertRaises(ValueError):h.patch_candidates(p,p,f,np.arange(len(f)))
if __name__=="__main__":unittest.main()

