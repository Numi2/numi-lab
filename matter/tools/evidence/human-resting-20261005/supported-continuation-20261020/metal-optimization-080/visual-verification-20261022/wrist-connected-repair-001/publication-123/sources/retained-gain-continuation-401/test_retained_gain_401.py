import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path("/Users/n/Work/numi-human-connected-surface-069/D")
sys.path.insert(0, str(ROOT / "precision-phase-one-396"))
import propose_joint_clearance_witnesses_396 as legacy396
sys.path.insert(0, str(ROOT / "retained-gain-continuation-401"))
import propose_joint_clearance_witnesses_401 as proposal401


class RetainedGainBudgetTests(unittest.TestCase):
    def test_95_percent_uses_fraction_of_measured_gain_not_absolute_fraction(self):
        receipt = proposal401.stage2_retained_gain_budget(
            0.2, np.array([0.4, 0.6]), 0.001,
            sparse.csr_matrix((0, 1)), np.empty(0),
            np.array([-1.0]), np.array([1.0]),
            retained_gain_fraction=0.95,
        )
        self.assertAlmostEqual(receipt["zero_move_baseline_objective"], 1.0, places=14)
        self.assertAlmostEqual(receipt["recognized_gain_vs_zero"], 0.8, places=14)
        self.assertAlmostEqual(receipt["effective_slack_increase"], 0.04, places=14)
        self.assertAlmostEqual(receipt["stage2_slack_budget"], 0.24, places=14)
        self.assertGreater(receipt["effective_slack_increase"], 0.001)
        self.assertFalse(receipt["relative_policy_uses_absolute_tolerance_cap"])

    def test_zero_gain_gives_no_relative_slack_back(self):
        receipt = proposal401.stage2_retained_gain_budget(
            1.0, np.array([0.4, 0.6]), 0.001,
            sparse.csr_matrix((0, 1)), np.empty(0),
            np.array([-1.0]), np.array([1.0]),
            retained_gain_fraction=0.95,
        )
        self.assertAlmostEqual(receipt["recognized_gain_vs_zero"], 0.0, places=14)
        self.assertAlmostEqual(receipt["effective_slack_increase"], 0.0, places=14)
        self.assertAlmostEqual(receipt["stage2_slack_budget"], 1.0, places=14)

    def test_absent_and_one_fraction_match_396_legacy_solver(self):
        args = (
            sparse.csr_matrix((0, 1)), np.empty(0), np.array([0.0]),
            np.array([-1.0]), np.array([1.0]), sparse.csr_matrix([[1.0]]),
            np.array([1.0]), np.array([1.0]), 5.0,
        )
        old = legacy396.solve_sparse_l1_clearance(*args, slack_tolerance=0.01)
        default = proposal401.solve_sparse_l1_clearance(*args, slack_tolerance=0.01)
        explicit_one = proposal401.solve_sparse_l1_clearance(
            *args, slack_tolerance=0.01, retained_gain_fraction=1.0
        )
        self.assertTrue(old["success"] and default["success"] and explicit_one["success"])
        np.testing.assert_allclose(default["delta"], old["delta"], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(explicit_one["delta"], old["delta"], rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(default["stage2_slack_budget_policy"]["stage2_slack_budget"], 0.01, places=10)
        self.assertAlmostEqual(explicit_one["stage2_slack_budget_policy"]["stage2_slack_budget"], 0.01, places=10)
        self.assertEqual(default["stage2_slack_budget_policy"]["mode"],
                         "legacy_absolute_slack_tolerance_parity")

    def test_relative_fraction_is_applied_in_solver_budget(self):
        args = (
            sparse.csr_matrix((0, 1)), np.empty(0), np.array([0.0]),
            np.array([-1.0]), np.array([1.0]), sparse.csr_matrix([[1.0]]),
            np.array([1.0]), np.array([1.0]), 5.0,
        )
        result = proposal401.solve_sparse_l1_clearance(
            *args, slack_tolerance=0.001, retained_gain_fraction=0.95
        )
        self.assertTrue(result["success"], result)
        self.assertAlmostEqual(result["stage2_slack_budget_policy"]["stage2_slack_budget"], 0.05, places=8)
        self.assertAlmostEqual(result["delta"][0], 0.95, places=7)

    def test_retained_gain_opt_in_is_incompatible_with_precision_phase_one(self):
        proposal401.validate_gain_retention_phase_one(None, True)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            proposal401.validate_gain_retention_phase_one(0.95, True)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            proposal401.validate_gain_retention_phase_one(1.0, True)
        proposal401.validate_gain_retention_phase_one(0.95, False)

    def test_nonfinite_or_out_of_range_fraction_fails_closed(self):
        for value in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    proposal401.stage2_retained_gain_budget(
                        0.2, np.array([1.0]), 1e-9,
                        sparse.csr_matrix((0, 1)), np.empty(0),
                        np.array([-1.0]), np.array([1.0]),
                        retained_gain_fraction=value,
                    )


if __name__ == "__main__":
    unittest.main()
