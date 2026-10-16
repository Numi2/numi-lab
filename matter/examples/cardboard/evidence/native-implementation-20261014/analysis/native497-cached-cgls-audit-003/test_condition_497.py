#!/usr/bin/env python3
"""Structural regression checks for the sealed native497 CPU result."""
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report-v2.json"


class DirectSolveAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_capture_scope_and_mask_are_explicit(self):
        r = self.report
        c = r["capture"]
        self.assertEqual(c["operator_dimension_free_xyz_only"], 2538)
        self.assertEqual(c["operator_nnz"], 71046)
        self.assertEqual(c["fixed_node_count"], 36)
        self.assertEqual(c["free_node_count"], 846)
        self.assertEqual(c["captured_cgls_selected_words"], [1, 0])
        self.assertTrue(c["fixed_xyz_dofs_eliminated"] == 108)
        self.assertTrue(c["field_backing_dofs_excluded"])
        self.assertTrue(c["rigid_contact_cohesive_unknowns_excluded"])

    def test_direct_solve_has_small_true_residual_and_backward_error(self):
        rows = self.report["sparse_direct_solve"]["solve_vectors"]
        x = rows["FP64_sparse_LU_solution"]
        self.assertTrue(x["finite"])
        self.assertLess(x["residual_over_rhs_l2"], 1.0e-10)
        self.assertLess(x["normwise_backward_error_1"], 1.0e-15)
        self.assertLess(x["componentwise_backward_error_inf"], 1.0e-12)

    def test_condition_number_is_estimate_not_bound(self):
        c = self.report["conditioning"]
        vals = c["condition_one_estimates"]
        self.assertEqual(len(vals), 3)
        self.assertTrue(all(math.isfinite(x) and x > 1.0 for x in vals))
        self.assertEqual(vals[0], vals[1])
        self.assertEqual(vals[1], vals[2])
        self.assertIn("not a certified upper bound", c["interpretation"])

    def test_native_warm_candidate_is_measured_against_same_direct_solution(self):
        rows = self.report["sparse_direct_solve"]["solve_vectors"]
        x0 = rows["captured_original_Newton_x0"]
        warm = rows["captured_total_warm_CGLS_X"]
        self.assertAlmostEqual(x0["residual_l2"], 2.095578359180791e-7, delta=1e-18)
        self.assertAlmostEqual(warm["residual_l2"], 2.6548842652915708e-8, delta=1e-18)
        self.assertGreater(x0["relative_forward_error_l2_vs_LU_solution"], 0.0)
        self.assertGreater(warm["relative_forward_error_l2_vs_LU_solution"], 0.0)

    def test_compensated_resummation_is_narrowly_scoped(self):
        c = self.report["assembly_and_precision"]["global_duplicate_entry_resummation"]
        self.assertIn("same FP64 local element entries", c["method"])
        self.assertLess(c["relative_stiffness_difference_l2"], 1.0e-12)
        self.assertLess(c["compensated_solution_relative_to_direct_l2"], 1.0e-10)
        self.assertIn("local tensor/geometry arithmetic remains binary64", c["scope_note"])

    def test_native_run_is_retained_as_failed_segment(self):
        run = self.report["native_run_binding"]
        self.assertEqual(run["exit_code"], 2)
        self.assertFalse(run["timed_out"])
        self.assertTrue(run["source_inputs_unchanged"])
        self.assertTrue(run["prediction_inputs_all_verified"])
        self.assertTrue(any("failed/incomplete" in x for x in self.report["limits"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
