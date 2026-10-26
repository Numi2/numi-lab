import unittest
import numpy as np

from fixed_bone_outward_plane_458 import (
    outward_plane_for_pair, build_fixed_bone_outward_plane_rows,
)


class FixedBoneOutwardPlane458Tests(unittest.TestCase):
    def test_plane_lower_bounds_separate_complete_triangle(self):
        # Outward normal is +x. Two skin corners lie behind the bone plane.
        bone = np.array([[0., -2., -2.], [0., 2., -2.], [0., 0., 2.]])
        skin = np.array([[0.3, -1., -1.], [-0.4, 1., -1.], [-0.1, 0., 1.]])
        plane = outward_plane_for_pair(skin, bone)
        n = plane["normal_world_unit"]
        self.assertGreater(n[0], .999999)
        self.assertGreater(plane["float32_ulp_margin_mm"], 0.)
        moved = skin + plane["corner_lower_mm"][:, None] * n[None, :]
        self.assertTrue(np.all(moved @ n >= bone[0] @ n + plane["float32_ulp_margin_mm"] - 1e-12))
        self.assertGreater(float(np.min(moved @ n) - np.max(bone @ n)), 0.)

    def test_all_three_bounds_include_negative_rhs_and_soft_rows_are_positive_only(self):
        points = np.array([[.10554, 0., 0.], [-.14272, 1., 0.], [-.002208, 0., 1.]])
        faces = np.array([[0, 1, 2]])
        bone = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        pair = dict(face_row=0, body=43, bone_id=16, triangle_id=133)
        result = build_fixed_bone_outward_plane_rows(
            points, faces, np.array([0, 1, 2]), np.array([], dtype=int),
            [pair], {(43, 16, 133): bone}, total_weight=.5)
        self.assertEqual(result["G"].shape, (3, 9))
        self.assertEqual(len(result["lower"]), 3)
        self.assertEqual(result["C"].shape[0], 2)
        self.assertAlmostEqual(float(result["weights"].sum()), .5, places=12)
        self.assertLess(result["lower"][0], 0.)
        self.assertGreater(result["lower"][1], 0.)
        # A displacement satisfying all three lowers puts the complete skin
        # triangle beyond the fixed bone triangle plane.
        displacement = np.zeros((3, 3))
        for row, meta in enumerate(result["guard_metadata"]):
            displacement[meta["corner_index"]] = (
                result["lower"][row] * np.asarray(meta["normal_world_unit"]))
        moved = points + displacement
        n = result["guard_metadata"][0]["normal_world_unit"]
        n = np.asarray(n)
        margin = result["guard_metadata"][0]["float32_ulp_margin_mm"]
        self.assertGreaterEqual(float(np.min((moved - bone[0]) @ n)), margin - 1e-12)

    def test_zero_anchor_coefficients_and_positive_anchor_conflict_reported(self):
        points = np.array([[.2, 0., 0.], [-.1, 1., 0.], [-.2, 0., 1.]])
        faces = np.array([[0, 1, 2]])
        bone = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        result = build_fixed_bone_outward_plane_rows(
            points, faces, np.array([0, 1, 2]), np.array([1]), [
                dict(face_row=0, body=1, bone_id=2, triangle_id=3)
            ], {(1, 2, 3): bone})
        self.assertEqual(result["G"].shape, (3, 9))
        self.assertEqual(result["G"][1].nnz, 0)
        self.assertEqual(result["C"].shape[0], 1)
        self.assertEqual(result["diagnostics"]["anchored_positive_lower_count"], 1)

    def test_verified_hand_example_signed_distances(self):
        expected_n = np.array([-.805945, -.584462, -.094107], dtype=np.float64)
        expected_n /= np.linalg.norm(expected_n)
        reference = np.array([0., 0., 1.])
        tangent1 = np.cross(expected_n, reference)
        tangent1 /= np.linalg.norm(tangent1)
        tangent2 = np.cross(expected_n, tangent1)
        bone = np.array([[0., 0., 0.], tangent1 * 10., tangent2 * 10.])
        distances = np.array([.10554, -.14272, -.002208])
        skin = np.array([
            distances[0] * expected_n,
            distances[1] * expected_n + tangent1,
            distances[2] * expected_n + tangent2,
        ])
        plane = outward_plane_for_pair(skin, bone)
        self.assertGreater(float(plane["normal_world_unit"] @ expected_n), .999999)
        np.testing.assert_allclose(plane["corner_signed_distance_mm"], distances, atol=1e-12)
        self.assertLess(plane["corner_lower_mm"][0], 0.)
        self.assertGreater(plane["corner_lower_mm"][1], 0.)
        self.assertGreater(plane["corner_lower_mm"][2], 0.)

    def test_outward_sign_follows_verified_triangle_order(self):
        skin = np.array([[.1, 0., 0.], [-.1, 1., 0.], [0., 0., 1.]])
        bone = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        normal = outward_plane_for_pair(skin, bone)["normal_world_unit"]
        self.assertGreater(normal[0], .999999)
        reversed_bone = bone[[0, 2, 1]]
        reversed_normal = outward_plane_for_pair(skin, reversed_bone)["normal_world_unit"]
        self.assertLess(reversed_normal[0], -.999999)


if __name__ == "__main__":
    unittest.main()

