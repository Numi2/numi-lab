import unittest,numpy as np
import star_candidates569 as s
class StarTests(unittest.TestCase):
 def setup_star(self,k):
  t=np.arange(k)*2*np.pi/k
  p=np.vstack((np.zeros((1,3)),np.column_stack((np.cos(t),np.sin(t),np.zeros(k)))))
  f=np.asarray([[0,i+1,(i+1)%k+1] for i in range(k)],dtype=np.int64)
  return p,f
 def test_ring_orientation(self):
  p,f=self.setup_star(6);self.assertEqual(s.ring_for_star(f,range(6),0),(1,2,3,4,5,6))
 def test_boundary_and_all_points(self):
  p,f=self.setup_star(7);c=s.star_candidates(p,p,f,range(7),0)
  self.assertTrue(c)
  def boundary(t):
   from collections import Counter
   z=Counter((int(a),int(b)) for tri in t for a,b in zip(tri,np.roll(tri,-1)))
   return sorted(e for e in z if z[e]-z[(e[1],e[0])]==1)
  for n in c:
   self.assertEqual(n.shape,f.shape);self.assertEqual(set(n.flat),set(f.flat))
   self.assertEqual(boundary(n),boundary(f))
   self.assertTrue(np.all(np.cross(p[n[:,1]]-p[n[:,0]],p[n[:,2]]-p[n[:,0]])[:,2]>0))
 def test_open_and_branches_reject(self):
  p,f=self.setup_star(6)
  with self.assertRaises(ValueError):s.ring_for_star(f,range(5),0)
  f[0]=f[1]
  with self.assertRaises(ValueError):s.ring_for_star(f,range(6),0)
 def test_convex_catalan(self):
  p,f=self.setup_star(6);xy,c,eps=s.chart(p,range(1,7),0)
  self.assertEqual(len(s.polygon_triangulations(xy,eps)),14)
 def test_external_edge_reject(self):
  p,f=self.setup_star(6);n=s.star_candidates(p,p,f,range(6),0)[0]
  oe={tuple(sorted((int(a),int(b))))for tri in f for a,b in zip(tri,np.roll(tri,-1))}
  ne={tuple(sorted((int(a),int(b))))for tri in n for a,b in zip(tri,np.roll(tri,-1))}
  e=next(iter(ne-oe));self.assertTrue(s.external_edge_conflict(f,range(6),n,{e:[(99,0)]}))
  self.assertFalse(s.external_edge_conflict(f,range(6),n,{}))
 def test_concave_triangulation_area(self):
  p=np.asarray([[0,0],[2,0],[2,2],[1,.8],[0,2]],float);eps=1e-12
  ts=s.polygon_triangulations(p,eps);self.assertTrue(ts)
  area=sum(s.orient(np.zeros(2),p[i],p[(i+1)%len(p)])for i in range(len(p)))
  for t in ts:self.assertAlmostEqual(sum(s.orient(p[a],p[b],p[c])for a,b,c in t),area)
if __name__=='__main__':unittest.main()

