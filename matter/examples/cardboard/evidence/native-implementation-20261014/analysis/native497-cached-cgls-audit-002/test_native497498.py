#!/usr/bin/env python3
"""Bounded regression checks over the sealed native497/498/482 report."""
import json
import unittest
from pathlib import Path

REPORT = Path(__file__).with_name("report-v4.json")


class NativeParityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_three_runs_are_clean_expected_terminal_failures(self):
        self.assertEqual(set(self.doc["receipt_summary"]), {"482", "497", "498"})
        for row in self.doc["receipt_summary"].values():
            self.assertEqual(row["exit_code"], 2)
            self.assertFalse(row["timed_out"])
            self.assertTrue(row["source_inputs_unchanged"])

    def test_physical_outputs_and_production_logs_are_exact(self):
        out = self.doc["output_comparison"]
        self.assertEqual(out["output_path_count_each"], 38)
        self.assertEqual(out["production_physical_file_count"], 35)
        self.assertTrue(out["production_physical_files_byte_identical_all_three"])
        self.assertEqual(out["differing_paths_vs_482"], {
            "497": ["coupon-resume-segment.json"],
            "498": ["coupon-resume-segment.json"],
        })
        self.assertTrue(self.doc["log_trace_parity"]["normalized_streams_byte_identical"])
        self.assertTrue(self.doc["log_trace_parity"]["retained_true_residual_and_CGLS_records_exact"])

    def test_trace_extent_and_observer_scope(self):
        traces = self.doc["log_trace_parity"]["trace_counts"]
        for run in traces.values():
            self.assertEqual(run, {"fem_true_residual_cycle": 4096, "fem_base_cgls_direction": 64})
        self.assertEqual(self.doc["log_trace_parity"]["observer_trace_counts_497"]["fem_residual_armijo_iteration"], 112)

    def test_source_revision_is_separate_from_observer_parity(self):
        scope = self.doc["protocol_and_scope"]
        self.assertTrue(scope["482_source_map_differs_from_497_498"])
        self.assertTrue(scope["497_498_same_frozen_source"])
        self.assertEqual(len(scope["482_to_497_source_changed_paths"]), 8)
        self.assertTrue(scope["normalized_solver_environment_equal_after_observer_controls_removed"])

    def test_residual_scale_and_no_false_condition_claim(self):
        m = self.doc["cached_operator_residual_analysis"]
        r = m["native_residual_comparison"]
        self.assertAlmostEqual(r["vector_difference_over_base_r0"], 0.023365662882566557, places=12)
        self.assertAlmostEqual(r["vector_difference_over_final_FP64_residual"], 0.18443206027756107, places=12)
        self.assertAlmostEqual(r["fresh_native_scalar_relative_error_vs_FP64"], 0.0019403426955053756, places=12)
        self.assertIn("no condition estimate", m["conditioning_context_from_prior_audit"]["condition_estimate_status"])
        self.assertFalse(self.doc["segment_outcome"]["trajectory_complete"])
        self.assertFalse(self.doc["segment_outcome"]["physical_qualification"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
