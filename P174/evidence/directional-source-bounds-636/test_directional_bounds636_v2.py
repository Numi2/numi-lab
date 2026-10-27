import unittest
import numpy as np
from itertools import product
from directional_bounds636_v2 import directional_bounds

class BoundsTests(unittest.TestCase):
 def test_randomized_boxes_and_old_cube(self):
  rng=np.random.default_rng(20261027); n=6000;source=rng.normal(size=(n,3))*100
  v=rng.normal(size=(n,3));v/=np.linalg.norm(v,axis=1)[:,None];dist=rng.uniform(0,12,size=n);p=source+v*dist[:,None]
  support=np.arange(n,dtype=np.int64);anchors=np.arange(0,n,337,dtype=np.int64)
  lo,hi,rec=directional_bounds(p,source,support,anchors,4,12);lo=lo.reshape(-1,3);hi=hi.reshape(-1,3)
  b=np.minimum(4,12-dist)/np.sqrt(3.);isfree=~np.isin(support,anchors)
  self.assertTrue(np.all(lo[isfree]<=-b[isfree,None]+1e-12));self.assertTrue(np.all(hi[isfree]>=b[isfree,None]-1e-12))
  self.assertTrue(np.array_equal(lo[anchors],np.zeros((len(anchors),3))));self.assertTrue(np.array_equal(hi[anchors],np.zeros((len(anchors),3))))
  for bits in product((0,1),repeat=3):
   d=np.where(np.asarray(bits)[None,:],hi,lo)
   self.assertTrue(np.all(np.linalg.norm(d,axis=1)<=4+1e-10));self.assertTrue(np.all(np.linalg.norm(p-source+d,axis=1)<=12+1e-10))
  self.assertGreater(rec['expanded_node_count'],0)
 def test_boundary_and_axis_sign(self):
  src=np.zeros((3,3));p=np.array([[12.,0,0],[-12.,0,0],[0,0,0.]])
  lo,hi,_=directional_bounds(p,src,np.arange(3),np.array([],dtype=int),4,12);lo=lo.reshape(-1,3);hi=hi.reshape(-1,3)
  self.assertLess(lo[0,0],-2);self.assertEqual(hi[0,0],0);self.assertGreater(hi[1,0],2);self.assertEqual(lo[1,0],0)
  self.assertAlmostEqual(lo[2,0],-4/np.sqrt(3));self.assertAlmostEqual(hi[2,0],4/np.sqrt(3))
 def test_invalid_inputs_fail_closed(self):
  for p,s,scope,anchors,t,c in [(np.array([[13.,0.,0.]]),np.zeros((1,3)),[0],[],4,12),
                              (np.zeros((1,3)),np.zeros((1,3)),[0,0],[],4,12),
                              (np.zeros((1,3)),np.zeros((1,3)),[1],[],4,12),
                              (np.zeros((1,3)),np.zeros((1,3)),[0],[1],4,12),
                              (np.zeros((1,3)),np.zeros((1,3)),[0],[],0,12),
                              (np.array([[0.,0.,0.],[13.,0.,0.]]),np.zeros((2,3)),[0],[],4,12)]:
   with self.assertRaises(ValueError):directional_bounds(p,s,scope,anchors,t,c)
if __name__=='__main__':unittest.main()
