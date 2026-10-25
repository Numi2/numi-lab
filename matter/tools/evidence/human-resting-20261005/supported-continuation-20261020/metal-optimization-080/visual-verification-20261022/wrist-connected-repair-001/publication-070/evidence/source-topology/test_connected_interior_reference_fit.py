import unittest
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"reference-fit-layer-counterfactual-001"))
from reference_fit_layer_counterfactual import N, tet_metrics
import connected_interior_reference_fit
from connected_interior_reference_fit import face_area_normals, fixed_boundary_gradient_fit

class ConnectedInteriorFitTests(unittest.TestCase):
    def test_face_area_and_normal(self):
        p=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        a,n=face_area_normals(p,np.array([[0,1,2]]))
        self.assertAlmostEqual(a[0],.5)
        np.testing.assert_allclose(n[0],[0,0,1])

    def test_fixed_boundary_gradient_extension_transports_constant_shift(self):
        # Small connected three-layer tetrahedral mesh; first three rows are O.
        X=np.random.default_rng(12).uniform(-1e-3,1e-3,size=(9,3))
        T=np.array([[0,1,2,3],[0,3,4,5],[0,5,6,7],[0,7,8,2]],dtype=np.int64)
        d=np.tile(np.array([[.2e-3,-.1e-3,.3e-3]]),(3,1))
        class H:
            @staticmethod
            def sparse_displacement_gradient(reference,tetrahedra,free):
                # Delegate to the pinned owner helper loaded by the real module.
                return connected_interior_reference_fit.load_repair_helper().sparse_displacement_gradient(reference,tetrahedra,free)
        # Exercise the core scaling formula directly with a local import of helper.
        import connected_interior_reference_fit as fit
        out,info=fit.fixed_boundary_gradient_fit(X,T,d,H,outer_count=3,iteration_limit=50)
        np.testing.assert_allclose(out[:3]-X[:3],d,atol=1e-14)
        np.testing.assert_allclose(out[3:]-X[3:],np.tile(d[:1],(6,1)),atol=1e-10)
        self.assertEqual(info["fixed_outer_max_error_m"],0.)

    def test_tet_quality_and_orientation(self):
        X=np.array([[0.,0,0],[1,0,0],[.5,np.sqrt(3)/2,0],[.5,np.sqrt(3)/6,np.sqrt(2/3)]])
        det,vol,q=tet_metrics(X,np.array([[0,1,2,3]],dtype=np.int64))
        self.assertGreater(det[0],0)
        self.assertAlmostEqual(q[0],1.0,places=14)

if __name__=="__main__":unittest.main()
