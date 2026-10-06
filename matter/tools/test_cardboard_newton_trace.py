#!/usr/bin/env python3
"""Synthetic integrity and alignment checks for cardboard_newton_trace.py."""
from __future__ import annotations

import json
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cardboard_newton_trace as trace


def iterate(root: int, iteration: int, stage: str, residual: float, alpha: float,
            *, cycle_accepted: bool = True, columns: int = 64, status: int = 0,
            arnoldi_residual: float = 0.01) -> dict:
    return {
        "schema": 1,
        "root": root,
        "environment_count": 1,
        "object_count": 1,
        "microtick": 0,
        "iteration": iteration,
        "stage": stage,
        "alpha": [alpha],
        "object_line_search": [[alpha, 0.99, 8, iteration]],
        "fgmres": [{
            "diagnostics": [arnoldi_residual, residual, 1.0 if cycle_accepted else 0.0, float(columns)],
            "nonlinear": [20.0, residual, 0.1, 0.0],
            "cycle_accepted": cycle_accepted,
        }],
        "status": [{"code": status, "fgmres_iterations_microstep_max": columns}],
    }


def terminal(root: int, last_iteration: int, residual: float, status: int = 0,
             validity_w: float = 1.0) -> dict:
    # Certificate residual is relative to max(final FGMRES normalizer, 1).
    return {
        "schema": 1,
        "root": root,
        "environment_count": 1,
        "object_count": 1,
        "microtick": 0,
        "last_iteration": last_iteration,
        "capture_order": "immediately_after_nm_mixed_certify_before_candidate_masks_or_commit",
        "objects": [{"nonlinear": [residual / 13.0, 0.0, 0.0, 0.0], "validity": [0.99, 0.0, 0.0, validity_w]}],
        "status": [{"code": status, "completed_microsteps_at_certificate": 0}],
        "fgmres": [{"diagnostics": [0.01, 13.0, 1.0, 64.0], "nonlinear": [20.0, 13.0, 0.1, 0.0]}],
    }


def log_for(*records: tuple[str, dict]) -> str:
    return "\n".join(prefix + json.dumps(value) for prefix, value in records) + "\n"


def full_stages(root: int, iteration: int, residual: float, alpha: float,
                **kwargs: object) -> list[tuple[str, dict]]:
    return [(trace.ITERATE_PREFIX, iterate(root, iteration, stage, residual, alpha, **kwargs))
            for stage in trace.STAGES]


class NewtonTraceTests(unittest.TestCase):
    def test_uses_next_assembly_for_residual_growth(self) -> None:
        records = full_stages(7, 0, 10.0, 0.95)
        records += full_stages(7, 1, 13.0, 0.4)
        records.append((trace.TERMINAL_PREFIX, terminal(7, 1, 12.0)))
        result = trace.analyze_trace(log_for(*records), expected_root=7)
        first, second = result["iterations"]
        self.assertEqual(first["post_step_residual_source"], "next_iteration_reassembled_nonlinear_y")
        self.assertEqual(first["post_step_residual_norm"], 13.0)
        self.assertAlmostEqual(first["residual_growth_ratio"], 1.3)
        self.assertTrue(first["descriptive_screen_candidate"])
        self.assertFalse(second["near_full_alpha"])
        self.assertEqual([item["iteration"] for item in result["screen"]["observations"]], [0])
        self.assertIn("not an independently re-evaluated b-Ax", result["contract"]["arnoldi"])

    def test_rejects_unaccepted_or_stale_arnoldi_state(self) -> None:
        records = full_stages(2, 0, 10.0, 0.99, cycle_accepted=False, columns=0,
                              arnoldi_residual=8.0)
        records += full_stages(2, 1, 15.0, 0.5)
        records.append((trace.TERMINAL_PREFIX, terminal(2, 1, 14.0)))
        result = trace.analyze_trace(log_for(*records), expected_root=2)
        first = result["iterations"][0]
        self.assertTrue(first["residual_grew"])
        self.assertFalse(first["final_cycle_accepted"])
        self.assertEqual(result["screen"]["observations"], [])

    def test_zero_final_restart_columns_do_not_hide_a_latched_accepted_cycle(self) -> None:
        records = full_stages(9, 0, 10.0, 0.99, cycle_accepted=True, columns=0)
        records += full_stages(9, 1, 12.0, 0.4)
        records.append((trace.TERMINAL_PREFIX, terminal(9, 1, 11.0)))
        result = trace.analyze_trace(log_for(*records), expected_root=9)
        first = result["iterations"][0]
        self.assertEqual(first["final_restart_columns_diagnostic_only"], 0.0)
        self.assertTrue(first["descriptive_screen_candidate"])

    def test_exact_stage_state_comparison_catches_small_norm_drift(self) -> None:
        records = full_stages(10, 0, 1.0e-6, 0.99)
        changed = next(payload for prefix, payload in records if payload["stage"] == "after_rigid_limit")
        changed["fgmres"][0]["nonlinear"][1] = 1.01e-6
        result = trace.analyze_trace(log_for(*records), expected_root=10)
        self.assertEqual(result["integrity"], "invalid")
        self.assertTrue(any(error["reason"] == "residual_norm_changed_between_line_search_stages"
                            for error in result["trace_errors"]))

    def test_raw_terminal_status_overrides_certificate_flag(self) -> None:
        records = full_stages(5, 0, 10.0, 0.99)
        records.append((trace.TERMINAL_PREFIX, terminal(5, 0, 15.0, status=10, validity_w=1.0)))
        result = trace.analyze_trace(log_for(*records), expected_root=5)
        self.assertEqual(result["terminal"]["disposition"], "rejected_at_certificate_status")
        self.assertEqual(result["terminal"]["certificate_validity_w_informational_only"], 1.0)
        self.assertFalse(result["iterations"][0]["descriptive_screen_candidate"])
        self.assertNotIn("accepted", result["terminal"]["disposition"])

    def test_malformed_or_truncated_record_is_invalid(self) -> None:
        good = full_stages(1, 0, 3.0, 1.0)[0]
        text = log_for(good) + trace.ITERATE_PREFIX + '{"schema":1,"root":1'
        result = trace.analyze_trace(text, expected_root=1)
        self.assertEqual(result["integrity"], "invalid")
        self.assertEqual(len(result["parse_errors"]), 1)

    def test_missing_stage_is_incomplete_not_silently_complete(self) -> None:
        records = full_stages(3, 0, 5.0, 0.8)
        records = [record for record in records if record[1].get("stage") != "after_rigid_limit"]
        result = trace.analyze_trace(log_for(*records), expected_root=3)
        self.assertEqual(result["integrity"], "incomplete")
        self.assertIn("after_rigid_limit", result["iterations"][0]["missing_stages"])

    def test_missing_terminal_certificate_is_incomplete(self) -> None:
        result = trace.analyze_trace(log_for(*full_stages(11, 0, 5.0, 0.8)), expected_root=11)
        self.assertEqual(result["integrity"], "incomplete")
        self.assertEqual(result["terminal"]["disposition"], "missing_terminal_certificate")

    def test_terminal_identity_must_match_last_captured_iteration(self) -> None:
        records = full_stages(12, 0, 5.0, 0.8)
        records.append((trace.TERMINAL_PREFIX, terminal(99, 1, 4.0)))
        result = trace.analyze_trace(log_for(*records), expected_root=None)
        self.assertEqual(result["integrity"], "invalid")
        self.assertTrue(any(error["reason"] == "terminal_identity_mismatch"
                            for error in result["trace_errors"]))

    def test_missing_whole_iteration_is_invalid(self) -> None:
        for indices in [(0, 2), (1, 2)]:
            records = []
            for index in indices:
                records += full_stages(12, index, 5.0, 0.8)
            records.append((trace.TERMINAL_PREFIX, terminal(12, 2, 4.0)))
            result = trace.analyze_trace(log_for(*records), expected_root=12)
            self.assertEqual(result["integrity"], "invalid")
            self.assertTrue(any(error["reason"] == "noncontiguous_newton_iterations"
                                for error in result["trace_errors"]))

    def test_duplicate_terminal_is_invalid(self) -> None:
        records = full_stages(12, 0, 5.0, 0.8)
        records += [(trace.TERMINAL_PREFIX, terminal(12, 0, 4.0))] * 2
        result = trace.analyze_trace(log_for(*records), expected_root=12)
        self.assertEqual(result["integrity"], "invalid")
        self.assertTrue(any(error["reason"] == "duplicate_terminal_certificates"
                            for error in result["trace_errors"]))

    def test_cli_binds_hashes_and_never_overwrites_an_existing_report(self) -> None:
        records = full_stages(13, 0, 5.0, 0.8)
        records.append((trace.TERMINAL_PREFIX, terminal(13, 0, 4.0)))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "trace.log"
            output_path = directory / "analysis.json"
            input_bytes = log_for(*records).encode("utf-8")
            input_path.write_bytes(input_bytes)
            self.assertEqual(trace.main([str(input_path), "--output", str(output_path), "--root", "13"]), 0)
            original = output_path.read_bytes()
            payload = json.loads(original)
            self.assertEqual(payload["evidence_binding"]["input_log_sha256"], hashlib.sha256(input_bytes).hexdigest())
            self.assertEqual(payload["evidence_binding"]["decoder_sha256"], hashlib.sha256(Path(trace.__file__).read_bytes()).hexdigest())
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trace.main([str(input_path), "--output", str(output_path), "--root", "13"])
            self.assertEqual(output_path.read_bytes(), original)

    def test_cli_retains_incomplete_and_skipped_reports_but_fails(self) -> None:
        inputs = [log_for(*full_stages(4, 0, 5.0, 0.8)),
                  'fem_newton_trace_skipped={"schema":1,"root":4,"reason":"node_cap"}\n']
        with tempfile.TemporaryDirectory() as temporary:
            for index, content in enumerate(inputs):
                input_path = Path(temporary) / f"trace-{index}.log"
                output_path = Path(temporary) / f"report-{index}.json"
                input_path.write_text(content)
                self.assertEqual(trace.main([str(input_path), "--output", str(output_path), "--root", "4"]), 2)
                self.assertIn(json.loads(output_path.read_text())["integrity"], ("incomplete", "skipped"))

    def test_skip_is_explicit(self) -> None:
        result = trace.analyze_trace('fem_newton_trace_skipped={"schema":1,"root":4,"reason":"node_cap"}\n', expected_root=4)
        self.assertEqual(result["integrity"], "skipped")
        self.assertEqual(result["skips"][0]["reason"], "node_cap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
