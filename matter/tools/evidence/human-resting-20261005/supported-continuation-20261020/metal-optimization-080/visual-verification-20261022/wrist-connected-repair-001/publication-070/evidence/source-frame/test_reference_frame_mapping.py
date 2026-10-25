import unittest
import numpy as np
from reference_frame_mapping import weighted_skin_jacobians, map_world_delta_to_vertex_position_delta, quat_matrix

class ReferenceFrameMappingTests(unittest.TestCase):
    def test_one_binding_rotated_scaled_roundtrip(self):
        q = np.array([0.0, 0.0, np.sin(.31), np.cos(.31)])
        A = quat_matrix(q)[None] * 1.4
        J = weighted_skin_jacobians(np.ones((1,1)), A)
        d = np.array([[.003, -.002, .004]])
        dp, cond, smin, residual = map_world_delta_to_vertex_position_delta(J,d)
        np.testing.assert_allclose(np.einsum('nij,nj->ni',J,dp),d,atol=1e-14)
        self.assertGreater(smin[0],0)
        self.assertGreaterEqual(cond[0],1)
        self.assertLess(np.max(np.abs(residual)),1e-14)

    def test_blended_bodies_require_weighted_jacobian(self):
        A=np.stack((np.eye(3),quat_matrix([0,np.sin(.4),0,np.cos(.4)])))
        J=weighted_skin_jacobians(np.array([[.35,.65]]),A)
        d=np.array([[.001,.002,-.003]])
        dp,_,_,_=map_world_delta_to_vertex_position_delta(J,d)
        np.testing.assert_allclose(np.einsum('nij,nj->ni',J,dp),d,atol=1e-14)
        dominant_only=A[1].T@d[0]
        self.assertGreater(np.linalg.norm(J[0]@dominant_only-d[0]),1e-5)

    def test_singular_blend_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'singular'):
            map_world_delta_to_vertex_position_delta(np.zeros((1,3,3)),np.ones((1,3)))

    def test_shape_and_nonfinite_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'shape'):
            weighted_skin_jacobians(np.ones((2,2)),np.ones((3,3,3)))
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            map_world_delta_to_vertex_position_delta(np.full((1,3,3),np.nan),np.ones((1,3)))

if __name__=='__main__':
    unittest.main()

