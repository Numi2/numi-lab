import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('geom', Path(__file__).parents[1] / 'tools/human_skin_patch_geometry.py')
geom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geom)


class HalfspaceClipTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        self.f = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])

    def area(self, x, f):
        p = x[f]
        return np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)/2

    def test_oblique_complement_partition_ancestry_and_orientation(self):
        n = np.array([1.,2,3]); d = .7
        original = self.area(self.x,self.f)
        sums = np.zeros(4)
        for less in (True,False):
            x,f,p,e,w = geom.clip_surface_halfspace(self.x,self.f,n,d,keep_less=less)
            np.testing.assert_array_equal(x[:4],self.x)
            np.testing.assert_allclose(x[4:],np.sum(self.x[e]*w[:,:,None],axis=1))
            np.testing.assert_allclose(w.sum(axis=1),1)
            np.testing.assert_allclose(x[4:]@n,d,atol=1e-15)
            self.assertTrue(np.all((x[f]@n-d)*(1 if less else -1)<=1e-15))
            self.assertTrue(np.all((x.astype(np.float32)[f]@n-d)*(1 if less else -1)<=2e-7))
            nr = np.cross(self.x[self.f[p,1]]-self.x[self.f[p,0]],self.x[self.f[p,2]]-self.x[self.f[p,0]])
            nc = np.cross(x[f[:,1]]-x[f[:,0]],x[f[:,2]]-x[f[:,0]])
            self.assertTrue(np.all(np.einsum('ij,ij->i',nr,nc)>0))
            sums += np.bincount(p,weights=self.area(x,f),minlength=4)
            signature = geom.surface_topology_signature(f)
            self.assertEqual(len(signature),1)
            self.assertEqual(len(signature[0]['boundary_loops']),1)
            self.assertEqual(signature[0]['genus'],0)
        np.testing.assert_allclose(sums,original,atol=1e-15)

    def test_shared_cut_ids_and_fixed_source_endpoint(self):
        x,f,p,e,w = geom.clip_surface_halfspace(self.x,self.f,[1,-1,0],0)
        self.assertEqual(len(e),1)
        np.testing.assert_array_equal(e,[[1,2]])
        self.assertGreaterEqual(np.count_nonzero(f==4),2)
        self.assertTrue(np.all(x[f,0]-x[f,1]<=1e-15))

    def test_empty_and_all_or_none(self):
        for d,count in ((2,4),(-1,0)):
            x,f,p,e,w = geom.clip_surface_halfspace(self.x,self.f,[1,0,0],d)
            self.assertEqual(len(f),count)
            self.assertEqual(len(e),0)
            np.testing.assert_array_equal(x,self.x)
        x,f,p,e,w = geom.clip_surface_halfspace(self.x,np.empty((0,3),dtype=int),[1,0,0],.5)
        self.assertEqual(f.shape,(0,3))

    def test_normal_scaling(self):
        a=geom.clip_surface_halfspace(self.x,self.f,[1,2,3],.7)
        b=geom.clip_surface_halfspace(self.x,self.f,[1e200,2e200,3e200],.7e200)
        for i,j in zip(a,b): np.testing.assert_allclose(i,j,atol=1e-15)

    def test_invalid_inputs(self):
        for n,d in (([0,0,0],0),([1,0,np.nan],0),([1,0,0],np.inf)):
            with self.assertRaises(ValueError): geom.clip_surface_halfspace(self.x,self.f,n,d)
        with self.assertRaises(ValueError): geom.clip_surface_halfspace(self.x,self.f.astype(float),[1,0,0],0)
        with self.assertRaises(ValueError): geom.clip_surface_halfspace(self.x,np.array([[0,1,10]]),[1,0,0],0)
        with self.assertRaises(ValueError): geom.clip_surface_halfspace(self.x,self.f,[1,0,0],0,keep_less='yes')

if __name__=='__main__': unittest.main()
