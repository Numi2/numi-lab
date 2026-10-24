import unittest
import numpy as np
from human_skin_patch_geometry import collapse_zero_length_surface_edges as collapse

class ZeroEdgeRepairTests(unittest.TestCase):
    def mesh(self):
        v=np.array([[0,0,1],[0,0,-1],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],np.float32)
        f=np.array([[0,2,3],[0,3,4],[0,4,5],[0,5,2],[1,3,2],[1,4,3],[1,5,4],[1,2,5]])
        return v,f
    def test_geometry_and_source_ancestry(self):
        v,f=self.mesh();v[0]=v[2];before=f.copy();g,a,r=collapse(v,f,[0])
        self.assertEqual(len(g),6);self.assertEqual(a[2],0)
        np.testing.assert_array_equal(v[g],v[f[r]]);np.testing.assert_array_equal(before,f)
        self.assertEqual(set(r),{1,2,4,5,6,7})
    def test_protected_destination(self):
        v,f=self.mesh();v[0]=v[2];g,a,r=collapse(v,f,[2]);self.assertEqual(a[0],2)
    def test_two_protected_rejected(self):
        v,f=self.mesh();v[0]=v[2]
        with self.assertRaisesRegex(ValueError,'protected'):collapse(v,f,[0,2])
    def test_boundary_edge_rejected(self):
        v,f=self.mesh();v[0]=v[2]
        with self.assertRaisesRegex(ValueError,'boundary'):collapse(v,f[1:])
    def test_no_nonlocal_coordinate_weld(self):
        v,f=self.mesh();v[0]=v[1];g,a,r=collapse(v,f)
        np.testing.assert_array_equal(g,f);np.testing.assert_array_equal(a,np.arange(len(v)))
    def test_tetrahedron_duplicate_faces_rejected(self):
        v=np.array([[0,0,0],[0,0,0],[0,1,0],[0,0,1]],float)
        f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        with self.assertRaisesRegex(ValueError,'duplicate'):collapse(v,f)
    def test_collinear_not_silently_deleted(self):
        with self.assertRaisesRegex(ValueError,'collinear'):collapse(np.array([[0,0,0],[1,0,0],[2,0,0.]]),np.array([[0,1,2]]))
    def test_disconnected_vertex_link_rejected(self):
        v,f=self.mesh()
        vv=np.concatenate([v, v[1:]+[5,0,0]])
        ff=f.copy();ff[ff>0]+=5
        with self.assertRaisesRegex(ValueError,'vertex link'):
            collapse(vv,np.concatenate([f,ff]))
    def test_invalid_inputs_and_identity(self):
        v,f=self.mesh();g,a,r=collapse(v,f);np.testing.assert_array_equal(g,f)
        for bad in [np.array([-1]),np.array([len(v)]),np.array([.5])]:
            with self.assertRaises(ValueError):collapse(v,f,bad)
        for bad in [np.array([[0,0,1]]),np.array([[0,1,99]]),f.astype(float)]:
            with self.assertRaises(ValueError):collapse(v,bad)
        v[0,0]=np.nan
        with self.assertRaises(ValueError):collapse(v,f)
if __name__=='__main__':unittest.main()
