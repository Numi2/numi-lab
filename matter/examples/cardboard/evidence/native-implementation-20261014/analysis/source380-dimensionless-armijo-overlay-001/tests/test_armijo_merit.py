from __future__ import annotations

import unittest
import math

from armijo_merit_reference import armijo_accept, frozen_row_scales, merit


class DimensionlessArmijoTests(unittest.TestCase):
    def setUp(self):
        # Scales mirror the disparity seen in the native trace: a stress-like
        # row has a much larger J-weighted state scale than complementarity.
        self.J = [[2.05537635e7, 0.0], [0.0, 1.20615267e2]]
        self.state_scales = [1.0e-8, 1.0e-8]
        self.rows = frozen_row_scales(self.J, self.state_scales)
        self.current = [4.16333634e-16, 1.69126791e-16]
        # A controlled trial models the observable pattern in trace388: a
        # low-weight stress residual rises slightly while FB residual falls.
        # Trial row components are deliberately labeled synthetic; the old
        # native trace only retained its raw max, not this per-row vector.
        self.trial = [5.55e-16, 8.0e-17]

    def test_dimensionless_merit_can_accept_when_raw_merit_is_noise_dominated(self):
        raw_current = max(abs(x) for x in self.current)
        raw_trial = max(abs(x) for x in self.trial)
        self.assertFalse(armijo_accept(raw_current, raw_trial, 1.0))
        scaled_current = merit(self.current, self.rows)
        scaled_trial = merit(self.trial, self.rows)
        self.assertGreater(scaled_current, scaled_trial)
        self.assertTrue(armijo_accept(scaled_current, scaled_trial, 1.0))

    def test_trace388_row_scales_expose_unit_mismatch(self):
        # Values are recomputed from exact native trace FP32 words by the
        # companion evidence script; row 6 owns raw max, row 12 owns scaled.
        row_scales = [
            1.02118752e-7, 1.03924182e-7, 1.06042935e-7,
            7.04887488e-7, 1.96508468e-5, 1.00277045e-7,
            2.05537635e-1, 1.11238664e-2, 4.98564630e-3,
            7.93989024e-4, 2.55429217e-2, 2.79151158e-3,
            1.20615267e-6, 1.99650719e-7, 1.00291225e-7,
        ]
        current = [
            -8.19310684e-22, 3.00923299e-23, -2.69573840e-22,
            7.34541554e-20, -3.53820303e-19, 2.57003310e-23,
            -4.16333634e-16, -8.67361738e-17, -8.67361738e-18,
            -2.16840434e-18, 6.93889390e-17, 1.73472348e-17,
            1.69126791e-16, 4.09971391e-22, -4.83325610e-23,
        ]
        self.assertEqual(max(range(len(current)), key=lambda i: abs(current[i])), 6)
        self.assertEqual(max(range(len(current)), key=lambda i: abs(current[i]) / row_scales[i]), 12)
        self.assertAlmostEqual(merit(current, row_scales), 1.4022005e-10,
                               delta=1e-16)

    def test_row_unit_rescaling_preserves_scaled_merit(self):
        factors = [1.0e-4, 2.0e3]
        r2 = [self.current[i] * factors[i] for i in range(2)]
        J2 = [[self.J[i][j] * factors[i] for j in range(2)] for i in range(2)]
        rows2 = frozen_row_scales(J2, self.state_scales)
        self.assertAlmostEqual(merit(self.current, self.rows),
                               merit(r2, rows2), places=20)

    def test_weights_are_frozen_for_all_trials(self):
        baseline = merit(self.trial, self.rows)
        # A changed trial Jacobian must not be used to reweight this trial.
        Jtrial = [[self.J[i][j] * [1.0e5, 1.0e-4][i] for j in range(2)]
                  for i in range(2)]
        trial_reweighted = merit(self.trial, frozen_row_scales(Jtrial, self.state_scales))
        self.assertNotAlmostEqual(baseline, trial_reweighted, places=14)
        self.assertTrue(armijo_accept(merit(self.current, self.rows),
                                      baseline, 1.0))

    def test_equal_or_worse_merit_never_accepts_stagnation(self):
        cur = merit(self.current, self.rows)
        self.assertFalse(armijo_accept(cur, cur, 1.0))
        self.assertFalse(armijo_accept(cur, cur * (1.0 + 1.0e-7), 0.5))
        self.assertFalse(armijo_accept(0.0, 0.0, 1.0))

    def test_bad_rows_or_nonfinite_trial_fail_closed(self):
        with self.assertRaises(ValueError):
            frozen_row_scales([[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0])
        self.assertEqual(merit([0.0, math.nan], [1.0, 1.0]), float("inf"))
        self.assertFalse(armijo_accept(1.0, float("nan"), 0.5))


if __name__ == "__main__":
    unittest.main()
