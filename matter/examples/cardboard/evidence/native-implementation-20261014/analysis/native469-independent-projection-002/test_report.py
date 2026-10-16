#!/usr/bin/env python3
"""Artifact checks for the sealed six-row MP projection witness."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run-sealed002"
EXPECTED_TETS = [750, 846, 1571, 1599, 1601, 1680]


class WitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((OUT / "report.json").read_text())

    def test_exact_rows_and_original_failures_are_retained(self):
        report = self.report
        self.assertEqual([r["tet"] for r in report["resolved_rows"]], EXPECTED_TETS)
        self.assertEqual([r["tet"] for r in report["original_fp64_failures_preserved"]], EXPECTED_TETS)
        self.assertEqual(report["aggregate"]["capture_center_reference_valid_rows_after_independent_fallback"], 1728)
        self.assertEqual(report["bindings"]["source166_snapshot_integrity"]["verified_file_count"], 787)

    def test_all_three_mp_precisions_meet_unchanged_gates_and_stability(self):
        for row in self.report["resolved_rows"]:
            self.assertTrue(row["high_precision_result"]["stability_gate_passes"])
            self.assertLess(row["high_precision_result"]["stability_max_dynamic_scaled_difference"],
                            row["high_precision_result"]["stability_gate_dynamic_scaled_difference_threshold"])
            for digits in ("70", "80", "110"):
                result = row["precision_results"][digits]
                self.assertTrue(result["passes_both_unchanged_gates"])
                self.assertLess(float(result["scaled_residual"]), 2e-12)
                self.assertLess(float(result["dynamic_scaled_correction"]), 2e-11)
                self.assertEqual(len(result["root_state_mp_decimal"]), 15)

    def test_native_comparisons_are_reported_without_promoting_fd_qualification(self):
        for row in self.report["resolved_rows"]:
            self.assertIn("projected_state_relative_l2", row["native_comparisons"])
            self.assertIn("stress_P_local_Pa", row["native_comparisons"])
            self.assertEqual(len(row["native_comparisons"]["cached_tangent"]["column_relative_l2"]), 9)
            self.assertIn("not a native finite-difference/noise-floor qualification",
                          row["native_comparisons"]["cached_tangent"]["interpretation"])

    def test_tet846_records_quotient_derivative_artifact(self):
        row = next(r for r in self.report["resolved_rows"] if r["tet"] == 846)
        diag = row["tet846_fb_quotient_jacobian_diagnosis"]
        self.assertEqual(float(diag["authored_unsimplified_d_fb_d_lambda"]), -1.0)
        self.assertGreater(float(diag["source166_complex_step_d_fb_d_lambda"]), 1e6)
        self.assertGreater(float(diag["rationalized_value_denominator"]), diag["source166_rationalization_branch_cutoff"])
        self.assertLess(float(diag["rationalized_value_denominator"]), 1.1e-14)

    def test_saved_state_archive_matches_110_digit_roots_after_float_export(self):
        roots = np.load(OUT / "mp_states.npz")
        for row in self.report["resolved_rows"]:
            expected = np.asarray([float(x) for x in row["precision_results"]["110"]["root_state_mp_decimal"]])
            self.assertTrue(np.array_equal(roots[f"tet_{row['tet']}"], expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
