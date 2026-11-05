import unittest
from collections import defaultdict
import numpy as np
from numi_human_compiler_tables import ArrayRows,CompactOrientedFaceUses,compact_export_source
CORNERS=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
class Tables(unittest.TestCase):
 def test_rows(self):
  x=np.arange(20).reshape(5,4);a=ArrayRows({'row':range(5),'integer':x[:,2],'weight':x[:,1].astype(float)/3})
  expected=[dict(row=i,integer=int(x[i,2]),weight=float(x[i,1]/3)) for i in range(5)]
  self.assertEqual([dict(r) for r in a],expected)
  self.assertEqual(a[-1].copy(),expected[-1]);self.assertEqual([dict(r) for r in a[1:4]],expected[1:4])
  with self.assertRaises(IndexError):a[5]
  with self.assertRaises(KeyError):a[0]['foreign']
  with self.assertRaises(ValueError):ArrayRows({'a':[1],'b':[1,2]})
 def test_faces(self):
  rng=np.random.default_rng(320)
  for case in range(32):
   p=rng.normal(size=(60,3));t=np.array([rng.choice(60,4,replace=False) for _ in range(12)])
   m=rng.choice([15,16],len(t));old=defaultdict(list)
   for cell,tet in enumerate(t):
    for corner in CORNERS:
     tri=tuple(int(tet[j]) for j in corner);old[tuple(sorted(tri))].append((cell,tri,int(m[cell])))
   new=CompactOrientedFaceUses(p,t,m,CORNERS)
   self.assertEqual(list(new),list(old))
   self.assertEqual(list((k,list(v)) for k,v in new.items()),list(old.items()))
   for k,v in old.items():self.assertEqual(list(new[k]),v)
 def test_explicit_material_interface(self):
  p=np.eye(5,3);t=np.array([[0,1,2,3],[0,2,1,4]])
  x=CompactOrientedFaceUses(p,t,[15,16],CORNERS)
  uses=list(x[(0,1,2)]);self.assertEqual(uses,[(0,(0,2,1),15),(1,(0,1,2),16)])
 def test_nonmanifold(self):
  with self.assertRaises(ValueError):CompactOrientedFaceUses(np.eye(6,3),np.array([[0,1,2,3],[0,2,1,4],[0,1,2,5]]),[15]*3,CORNERS)
 def test_fail_closed_source(self):
  with self.assertRaises(ValueError):compact_export_source('def export(): pass')
if __name__=='__main__':unittest.main()
