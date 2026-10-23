from __future__ import annotations

import unittest

from matter.tools.cardboard_opencourant_overlay_measure import _integrate_quad_elastic


class CardboardOpenCourantOverlayMeasureTests(unittest.TestCase):
    def test_two_by_two_gauss_energy_captures_bilinear_gap_variance(self) -> None:
        # Unit square; normal separation varies linearly from 0.1 to 0.3 mm
        # across x. The exact area mean is 0.2 mm, while mean(delta**2) is
        # 0.04 + 0.01/3, so averaging before squaring would lose the variance.
        points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                  (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        separation = [(0.0, 0.0, 0.1), (0.0, 0.0, 0.3),
                      (0.0, 0.0, 0.3), (0.0, 0.0, 0.1)]
        identity_x = (1.0, 0.0, 0.0)
        identity_y = (0.0, 1.0, 0.0)
        normal = (0.0, 0.0, 1.0)

        result = _integrate_quad_elastic(
            points, separation, 100.0, 100.0, identity_x, identity_y, normal)

        exact = 0.5 * 100.0 * (0.04 + 0.01 / 3.0)
        mean_gap_squared = 0.5 * 100.0 * (0.2 ** 2)
        self.assertAlmostEqual(result["elastic_energy_N_mm"], exact, places=12)
        self.assertGreater(result["elastic_energy_N_mm"], mean_gap_squared)
        self.assertAlmostEqual(result["reference_area_mm2"], 1.0, places=12)

    def test_consistent_nodal_forces_are_equal_and_opposite(self) -> None:
        points = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                  (2.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        separation = [(0.01, 0.02, 0.03)] * 4
        result = _integrate_quad_elastic(
            points, separation, 100.0, 100.0,
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

        for axis in range(3):
            upper = sum(node[axis] for node in result["upper_corner_forces_N"])
            lower = sum(node[axis] for node in result["lower_corner_forces_N"])
            self.assertAlmostEqual(upper + lower, 0.0, places=12)
        self.assertAlmostEqual(result["reference_area_mm2"], 2.0, places=12)
        self.assertGreater(result["elastic_energy_N_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
