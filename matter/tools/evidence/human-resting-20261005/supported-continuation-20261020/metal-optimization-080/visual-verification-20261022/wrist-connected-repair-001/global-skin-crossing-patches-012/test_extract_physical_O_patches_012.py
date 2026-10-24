import sys, unittest
from pathlib import Path
import numpy as np
p=Path(__file__).with_name("extract_physical_O_patches_012.py")
sys.path.insert(0,str(p.parent))
import extract_physical_O_patches_012 as m
class PatchExtractionTests(unittest.TestCase):
    def test_edge_connected_seed_and_two_ring_expansion(self):
        f=np.array([[0,1,2],[1,3,2],[3,4,2],[5,6,7]],dtype=np.int64)
        a=m.adjacency(f);lev,cs=m.expand(np.array([0]),a,2)
        self.assertEqual([x.tolist() for x in cs],[[0,1,2]])
        self.assertEqual(lev[:3].tolist(),[0,1,2]);self.assertEqual(int(lev[3]),-1)
    def test_multiple_seeds_merge_when_rings_connect(self):
        f=np.array([[0,1,2],[1,3,2],[3,4,2],[4,5,2]],dtype=np.int64)
        a=m.adjacency(f);lev,cs=m.expand(np.array([0,3]),a,1)
        self.assertEqual(len(cs),1);self.assertEqual(cs[0].tolist(),[0,1,2,3])
    def test_hit_barycentric_roundtrip(self):
        t=np.array([[0.,0,0],[2,0,0],[0,3,0]])
        x=np.array([.5,.25,.25])@t;b,res=m.barycentric(x,t)
        np.testing.assert_allclose(b,[.5,.25,.25],atol=1e-14);self.assertLess(res,1e-14)
    def test_surface_mapping_direct_and_barycentric(self):
        x=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
        f=np.array([[0,1,2],[1,3,2]])
        q=np.array([[0,0,0],[.2,.3,0],[1,1,0]],float)
        vi,vd,fi,b,d,stable,code=m.map_surface(q,x,f)
        self.assertEqual(code.tolist(),[1,2,1]);np.testing.assert_allclose(d,0,atol=1e-14)
        self.assertTrue(stable.all());np.testing.assert_allclose(b[1].sum(),1,atol=1e-14)
if __name__=="__main__":unittest.main(verbosity=2)

