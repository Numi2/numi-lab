import json
import pathlib
import struct
import tempfile
import unittest

from trace_health import (
    TraceError, adjudicate_capture, adjudicate_sequence, parse_armijo_log,
    read_capture_status,
)


def row(iteration, solver=0, trial=0, accepted=True, base=1.0):
    return {
        "schema": 1,
        "root": 77,
        "newton_iteration": iteration,
        "environments": [{
            "base_l2": base,
            "trial_l2": base * 0.5 if accepted else None,
            "accepted": accepted,
            "selected_alpha": 1.0 if accepted else 0.0,
            "environment_line_search_alpha": 1.0 if accepted else 0.0,
            "solver_status_code": solver,
            "solver_status": {"code": solver},
            "trial_status": {"code": trial, "failing_index": 9,
                             "diagnostics": [1, 2, 3, 4]},
        }],
    }


class TraceHealthTests(unittest.TestCase):
    def test_healthy_sequence_does_not_treat_stale_trial_status_as_fatal(self):
        records = [row(0, solver=0, trial=10, accepted=True), row(1)]
        result = adjudicate_sequence(records, expected_root=77)
        self.assertIsNone(result["first_fatal_iteration"])
        self.assertTrue(result["rows"][0]["base_is_valid_center"])

    def test_private_candidate_reject_does_not_fatal_solver_sequence(self):
        records = [row(0, solver=0, trial=6, accepted=False), row(1)]
        result = adjudicate_sequence(records, expected_root=77)
        self.assertIsNone(result["first_fatal_iteration"])
        capture = {"root": 77, "iteration": 0}
        decision = adjudicate_capture(
            result, capture, capture_trial_status={"code": 6})
        self.assertFalse(decision["center_trial_comparison_allowed"])
        self.assertTrue(decision["center_base_valid"])

    def test_first_fatal_row_has_healthy_predecision_base_but_no_trial_pair(self):
        records = [row(0), row(1, solver=10, trial=6, accepted=False)]
        result = adjudicate_sequence(records, expected_root=77)
        self.assertEqual(result["first_fatal_iteration"], 1)
        self.assertTrue(result["rows"][1]["base_is_valid_center"])
        decision = adjudicate_capture(result, {"root": 77, "iteration": 1},
                                      capture_trial_status={"code": 6})
        self.assertFalse(decision["center_trial_comparison_allowed"])
        self.assertIn("selected_newton_is_first_fatal_iteration; its trial is not an accepted residual pair",
                      decision["reasons"])

    def test_postfatal_center_is_invalid_even_if_capture_trial_status_is_zero(self):
        records = [row(0), row(1, solver=10, trial=6, accepted=False), row(2)]
        result = adjudicate_sequence(records, expected_root=77)
        self.assertFalse(result["rows"][2]["base_is_valid_center"])
        decision = adjudicate_capture(result, {"root": 77, "iteration": 2},
                                      capture_trial_status={"code": 0})
        self.assertFalse(decision["center_trial_comparison_allowed"])
        self.assertIn("selected_newton_is_after_first_fatal_status", decision["reasons"])

    def test_missing_and_reordered_newton_records_reject(self):
        with self.assertRaisesRegex(TraceError, "missing, duplicated, or reordered"):
            adjudicate_sequence([row(0), row(2)])
        with self.assertRaisesRegex(TraceError, "missing, duplicated, or reordered"):
            adjudicate_sequence([row(1), row(0)])

    def test_capture_requires_exact_trace_root_and_existing_iteration(self):
        sequence = adjudicate_sequence([row(0), row(1)])
        with self.assertRaisesRegex(TraceError, "root does not match"):
            adjudicate_capture(sequence, {"root": 78, "iteration": 0})
        with self.assertRaisesRegex(TraceError, "absent from Armijo trace"):
            adjudicate_capture(sequence, {"root": 77, "iteration": 2})

    def test_status_reader_rejects_boolean_or_overflow_extent(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            payload = bytearray(48)
            (d / "sample.bin").write_bytes(payload)
            meta = {
                "schema": "numi.matter.coupled_armijo_trial_capture.v1",
                "payload": {"file": "sample.bin", "byte_count": 48,
                            "sha256": hashlib.sha256(payload).hexdigest()},
                "arrays": [{"name": "fem_armijo_statuses",
                            "phase": "selected", "dtype": "record:NMMatterStatusGPU",
                            "count": 1, "element_bytes": 48, "byte_count": 48,
                            "offset": True}],
            }
            manifest = d / "sample.json"
            manifest.write_text(json.dumps(meta))
            with self.assertRaisesRegex(TraceError, "extent"):
                read_capture_status(manifest, "selected")
            meta["arrays"][0]["offset"] = 16
            manifest.write_text(json.dumps(meta))
            with self.assertRaisesRegex(TraceError, "extent"):
                read_capture_status(manifest, "selected")

    def test_actual_payload_status_reader_checks_adjacent_payload_and_code(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            payload = bytearray(48)
            struct.pack_into("<8I", payload, 0, 6, 0, 0, 331, 0, 0, 0, 0)
            struct.pack_into("<4f", payload, 32, 88, 127, 0, 0)
            (d / "sample.bin").write_bytes(payload)
            import hashlib
            meta = {
                "schema": "numi.matter.coupled_armijo_trial_capture.v1",
                "payload": {"file": "/wrong/absolute/path/sample.bin",
                            "byte_count": 48,
                            "sha256": hashlib.sha256(payload).hexdigest()},
                "arrays": [{"name": "fem_armijo_statuses",
                            "phase": "armijo_measurement_before_decision",
                            "dtype": "record:NMMatterStatusGPU", "count": 1,
                            "element_bytes": 48, "byte_count": 48,
                            "offset": 0}],
            }
            manifest = d / "sample.json"
            manifest.write_text(json.dumps(meta))
            status = read_capture_status(
                manifest, "armijo_measurement_before_decision")
            self.assertEqual(status["code"], 6)
            self.assertEqual(status["failing_index"], 331)

    def test_log_parser_retains_all_rows_and_rejects_bad_json(self):
        source = (b"noise\n" + b"fem_residual_armijo_iteration=" +
                  json.dumps(row(0)).encode() + b"\n" +
                  b"fem_residual_armijo_iteration={bad}\n")
        with self.assertRaisesRegex(TraceError, "invalid JSON"):
            parse_armijo_log(source)
        valid = b"fem_residual_armijo_iteration=" + json.dumps(row(0)).encode()
        self.assertEqual(len(parse_armijo_log(valid)), 1)


if __name__ == "__main__":
    unittest.main()
