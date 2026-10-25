import unittest
import numpy as np
from human_skin_fat_connected_repair import cell_values_gradients

class CellQualityConventionTests(unittest.TestCase):
    def test_positive_mean_ratio_does_not_satisfy_volume_edge_quality_floor(self):
        x = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,8e-6]], dtype=np.float32)
        t = np.array([[0,1,2,3]], dtype=np.int64)
        J, _, q, _ = cell_values_gradients(x, x, t)
        self.assertEqual(float(J[0]), 1.0)
        expected = float(x[3,2]) / (2.0 ** 1.5)
        self.assertAlmostEqual(float(q[0]), expected, places=15)
        self.assertLess(float(q[0]), 1e-5)
        # Independent alternate metric: it passes the same numerical threshold,
        # which is why it must never be substituted for the owner's gate.
        det = float(x[3,2])
        sum_edge_sq = 6.0 + 3.0 * det * det
        mean_ratio = 12.0 * (det / 2.0) ** (2.0 / 3.0) / sum_edge_sq
        self.assertGreater(mean_ratio, 1e-5)

    def test_volume_edge_quality_is_scale_invariant_and_signed(self):
        x = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.125]], dtype=np.float32)
        t = np.array([[0,1,2,3]], dtype=np.int64)
        _, _, q, _ = cell_values_gradients(x, x, t)
        y = x * np.float32(8.0)
        _, _, scaled, _ = cell_values_gradients(y, y, t)
        np.testing.assert_array_equal(scaled, q)
        reflected = x.copy()
        reflected[:,2] *= -1
        J, _, negative, _ = cell_values_gradients(reflected, x, t)
        self.assertEqual(float(J[0]), -1.)
        np.testing.assert_array_equal(negative, -q)

if __name__ == '__main__':
    unittest.main()
