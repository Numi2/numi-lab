import importlib.util
import pathlib
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("postproject_self_trial_177", HERE / "postproject_self_trial_177.py")
OWNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWNER)


def row(nodes, coeffs, lower):
    return {"nodes": np.asarray(nodes, dtype=np.int64),
            "coefficients": np.asarray(coeffs, dtype=np.float64),
            "lower": float(lower)}


class PostprojectTests(unittest.TestCase):
    def test_uses_self_projection_instead_of_infeasible_projection_average(self):
        zs = np.array([[0.1, 0, 0], [0, 0, 0]], dtype=np.float64)
        zn = np.array([[-0.3, 0, 0], [0, 0, 0]], dtype=np.float64)
        za = np.array([[-0.3, 0, 0], [0, 0, 0]], dtype=np.float64)
        averaged = (zs + zn + za) / 3
        self.assertLess(averaged[0, 0], 0.0)
        proposal, report = OWNER.postproject_self_trial(
            zs, [], [row([0], [[1, 0, 0]], 0.08)], 1e-6)
        np.testing.assert_array_equal(proposal, zs)
        self.assertEqual(report["status"], "SELF_PROJECTION_SLACK_AND_ANCHORS_PASS")
        self.assertLess(report["max_self_linear_violation_mm"], 1e-12)
        self.assertTrue(report["exact_emitted_f32_self_audit_required"])
        self.assertFalse(report["geometry_admitted"])

    def test_anchor_clamp_preserves_unrelated_self_row(self):
        zs = np.array([[0.2, 0, 0], [0.4, 0, 0]], dtype=np.float64)
        proposal, report = OWNER.postproject_self_trial(
            zs, [0], [row([1], [[1, 0, 0]], 0.3)], 1e-6)
        np.testing.assert_array_equal(proposal[0], np.zeros(3))
        self.assertEqual(proposal[1, 0], 0.4)
        self.assertTrue(report["anchors_unchanged"])
        self.assertTrue(report["self_projection_slack_gate_pass"])

    def test_anchor_clamp_that_breaks_self_row_fails_closed(self):
        zs = np.array([[0.8, 0, 0], [0.4, 0, 0]], dtype=np.float64)
        proposal, report = OWNER.postproject_self_trial(
            zs, [0], [row([0, 1], [[1, 0, 0], [1, 0, 0]], 1.0)], 0.01)
        np.testing.assert_array_equal(proposal[0], np.zeros(3))
        self.assertFalse(report["self_projection_slack_gate_pass"])
        self.assertEqual(report["status"], "REJECT_SELF_PROJECTION_SLACK_OR_ANCHOR_GATE")
        self.assertAlmostEqual(report["max_self_linear_violation_mm"], 0.6)

    def test_nonzero_fixed_anchor_values_are_preserved(self):
        zs = np.zeros((2, 3), dtype=np.float64)
        values = np.array([[0.25, -0.5, 1.0]])
        proposal, report = OWNER.postproject_self_trial(
            zs, [1], [row([0], [[1, 0, 0]], -1.0)], 1e-5,
            anchor_values_mm=values)
        np.testing.assert_array_equal(proposal[1], values[0])
        self.assertTrue(report["anchors_unchanged"])

    def test_invalid_margin_and_anchor_inputs_rejected(self):
        zs = np.zeros((2, 3), dtype=np.float64)
        with self.assertRaises(ValueError):
            OWNER.postproject_self_trial(zs, [2], [], 1e-5)
        with self.assertRaises(ValueError):
            OWNER.postproject_self_trial(zs, [0, 0], [], 1e-5)
        with self.assertRaises(ValueError):
            OWNER.postproject_self_trial(zs, [], [], 0.0)

    def test_malformed_pair_rows_fail_closed(self):
        zs = np.zeros((2, 3), dtype=np.float64)
        with self.assertRaises(ValueError):
            OWNER.postproject_self_trial(zs, [], [row([0, 0], [[1, 0, 0], [-1, 0, 0]], 0)], 1e-4)
        with self.assertRaises(ValueError):
            OWNER.postproject_self_trial(zs, [], [row([1], [[1, 0]], 0)], 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
