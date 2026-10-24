import unittest
from unittest.mock import patch
import numpy as np
import human_skin_p1_obstacles as owner

class RobustConstructionClipTests(unittest.TestCase):
    def test_float32_thin_parent_cancellation(self):
        X=np.array([[.1,-.25,.17],[.11,-.25,.17],
                    [.105,-.249999985,.17],[.1,-.25,.18]],np.float32)
        P=np.array([[.11633384972810745,-.24418513476848602,.1711786538362503],
                    [.10825975984334946,-.25390860438346863,.16425739228725433],
                    [.10650855302810669,-.2451026439666748,.17971591651439667]],np.float32)
        before=X.copy(),P.copy()
        with patch.object(owner,"_exact_triangle_tet_polygon",
                          wraps=owner._exact_triangle_tet_polygon) as fallback:
            B,C=owner.triangle_tet_barycentric_polygon(X,P)
            self.assertEqual(fallback.call_count,1)
        self.assertEqual(len(B),4)
        self.assertTrue(np.all(B>=0));self.assertTrue(np.all(C>=0))
        np.testing.assert_allclose(B.sum(1),1,atol=3e-16,rtol=0)
        np.testing.assert_allclose(C.sum(1),1,atol=3e-16,rtol=0)
        np.testing.assert_allclose(B@X.astype(float),C@P.astype(float),atol=1e-16,rtol=0)
        np.testing.assert_array_equal(X,before[0]);np.testing.assert_array_equal(P,before[1])

    def test_retained_native_mesh_failure_witness(self):
        # 296b: tet97486; registered soft-tissue stable120 local face65.
        # Exact Float32 inputs retained independently in the witness receipt.
        X=np.array([[.1758553832769394,-.7473028898239136,.0795469731092453],
                    [.16101577877998352,-.7075363397598267,.07082080841064453],
                    [.18758445978164673,-.7528747320175171,.07155054807662964],
                    [.14928670227527618,-.7019645571708679,.07881723344326019]],np.float32)
        P=np.array([[.14608509838581085,-.6928191184997559,.07950963824987411],
                    [.15397833287715912,-.7118531465530396,.0771523043513298],
                    [.15532177686691284,-.7209075093269348,.07796395570039749]],np.float32)
        with patch.object(owner,"_exact_triangle_tet_polygon",
                          wraps=owner._exact_triangle_tet_polygon) as fallback:
            B,C=owner.triangle_tet_barycentric_polygon(X,P)
            self.assertEqual(fallback.call_count,1)
        self.assertEqual(B.shape,(4,4));self.assertEqual(C.shape,(4,3))
        self.assertTrue(np.all(B>=0));self.assertTrue(np.all(C>=0))
        np.testing.assert_allclose(B.sum(1),1,atol=3e-16,rtol=0)
        np.testing.assert_allclose(C.sum(1),1,atol=3e-16,rtol=0)
        np.testing.assert_allclose(B@X.astype(float),C@P.astype(float),atol=2e-16,rtol=0)

    def test_regular_clip_retains_fast_path(self):
        X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        P=np.array([[-1,.2,.2],[1,.2,.2],[0,.8,.2]])
        with patch.object(owner,"_exact_triangle_tet_polygon",side_effect=AssertionError("unnecessary fallback")):
            B,C=owner.triangle_tet_barycentric_polygon(X,P)
        self.assertGreaterEqual(len(B),3)
        np.testing.assert_allclose(B@X,C@P,atol=1e-15)

    def test_exact_face_touch_and_empty(self):
        X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        B,C=owner._exact_triangle_tet_polygon(X,X[:3])
        self.assertEqual(len(B),3)
        np.testing.assert_array_equal(B@X,C@X[:3])
        B,C=owner._exact_triangle_tet_polygon(X,X[:3]+3)
        self.assertEqual(B.shape,(0,4));self.assertEqual(C.shape,(0,3))

    def test_reversed_parent_orientation_is_same_polygon(self):
        X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        P=np.array([[-1,.2,.2],[1,.2,.2],[0,.8,.2]])
        B,C=owner._exact_triangle_tet_polygon(X,P)
        Y=X[[0,2,1,3]]
        D,F=owner._exact_triangle_tet_polygon(Y,P)
        first=sorted(map(tuple,B@X));second=sorted(map(tuple,D@Y))
        np.testing.assert_allclose(first,second,atol=1e-15)
        np.testing.assert_allclose(F@P,D@Y,atol=1e-15)

    def test_invalid_condition_gate_unchanged(self):
        X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0]],float)
        with self.assertRaisesRegex(ValueError,"ill-conditioned"):
            owner.triangle_tet_barycentric_polygon(X,X[:3])

if __name__=="__main__":unittest.main()
