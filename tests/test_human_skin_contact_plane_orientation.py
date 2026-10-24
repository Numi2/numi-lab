import unittest

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from human_skin_contact_planes import assemble_barycentric_plane_row


class ContactPlaneRowTests(unittest.TestCase):
    def setUp(self):
        # Face IDs are intentionally supplied in unsorted order.
        self.face_a, self.face_b = 19, 3
        self.vertices_a = np.array([8, 1, 6], dtype=np.int64)
        self.vertices_b = np.array([4, 9, 2], dtype=np.int64)
        self.points = np.array([
            [0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.1, 0.0],
            [0.2, 0.2, 0.004], [0.8, 0.2, 0.004], [0.1, 0.9, 0.004],
            [0.9, 0.1, 0.0],
        ], dtype=np.float64)
        self.maps = np.repeat(np.eye(3)[None, :, :], len(self.points), axis=0)
        # Add distinct small rotations so the Jacobian is not just a sign test.
        self.maps[1] = [[1.0, .02, 0], [-.01, 1, .03], [0, -.02, .99]]
        self.maps[8] = [[.97, 0, -.04], [.02, 1.01, 0], [.05, 0, 1.0]]
        self.bary_a = np.array([.2, .3, .5])
        self.bary_b = np.array([.4, .35, .25])
        pa = self.bary_a @ self.points[self.vertices_a]
        pb = self.bary_b @ self.points[self.vertices_b]
        self.normal = (pa - pb) / np.linalg.norm(pa - pb)
        # Free variable order differs from vertex ID order; vertex 4 is fixed.
        self.free = {8: 2, 1: 0, 6: 3, 9: 1, 2: 4}

    def assemble(self, reverse=False):
        if reverse:
            return assemble_barycentric_plane_row(
                self.face_b, self.face_a, self.vertices_b, self.vertices_a,
                self.bary_b, self.bary_a, -self.normal, self.points,
                self.maps, self.free,
            )
        return assemble_barycentric_plane_row(
            self.face_a, self.face_b, self.vertices_a, self.vertices_b,
            self.bary_a, self.bary_b, self.normal, self.points,
            self.maps, self.free,
        )

    def test_unsorted_face_ids_are_canonicalized(self):
        row = self.assemble()
        self.assertEqual(row["face_ids"], (3, 19))
        np.testing.assert_array_equal(row["bary_a"], self.bary_b)
        np.testing.assert_array_equal(row["bary_b"], self.bary_a)
        np.testing.assert_array_equal(row["normal"], -self.normal)

    def test_pair_reversal_preserves_gap_and_jacobian(self):
        direct = self.assemble()
        reversed_row = self.assemble(reverse=True)
        self.assertEqual(direct["face_ids"], reversed_row["face_ids"])
        self.assertAlmostEqual(direct["gap_mm"], reversed_row["gap_mm"], places=12)
        np.testing.assert_allclose(direct["normal"], reversed_row["normal"], atol=0, rtol=0)
        np.testing.assert_allclose(direct["bary_a"], reversed_row["bary_a"], atol=0, rtol=0)
        np.testing.assert_allclose(direct["bary_b"], reversed_row["bary_b"], atol=0, rtol=0)
        self.assertEqual(set(direct["coeff"]), set(reversed_row["coeff"]))
        for key in direct["coeff"]:
            self.assertAlmostEqual(direct["coeff"][key], reversed_row["coeff"][key], places=12)

    def test_jacobian_matches_finite_difference_in_mm_variables(self):
        row = self.assemble()
        rng = np.random.default_rng(8127)
        d = rng.normal(size=3 * len(self.free))
        d *= 0.03 / np.linalg.norm(d)
        predicted_mm = sum(row["coeff"].get(k, 0.0) * d[k] for k in range(len(d)))
        # Gap in millimetres after mapping a millimetre displacement through D.
        points_plus = self.points.copy()
        points_minus = self.points.copy()
        for vertex, local in self.free.items():
            dv_m = self.maps[vertex] @ d[3*local:3*local+3] * 0.001
            points_plus[vertex] += dv_m
            points_minus[vertex] -= dv_m
        plus = assemble_barycentric_plane_row(
            self.face_a, self.face_b, self.vertices_a, self.vertices_b,
            self.bary_a, self.bary_b, self.normal, points_plus, self.maps, self.free,
        )["gap_mm"]
        minus = assemble_barycentric_plane_row(
            self.face_a, self.face_b, self.vertices_a, self.vertices_b,
            self.bary_a, self.bary_b, self.normal, points_minus, self.maps, self.free,
        )["gap_mm"]
        finite_difference_mm = (plus - minus) / 2.0
        self.assertAlmostEqual(predicted_mm, finite_difference_mm, places=10)


if __name__ == "__main__":
    unittest.main()
