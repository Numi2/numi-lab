import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cardboard_diagnostic_history_compare as compare


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value) -> bytes:
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


class DiagnosticHistoryComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.left = self.root / "diagnostic"
        self.right = self.root / "control"
        self._make_attempt(self.left, enabled=True)
        self._make_attempt(self.right, enabled=False)

    def tearDown(self):
        self.temp.cleanup()

    def _make_attempt(self, case: Path, enabled: bool):
        source = b"immutable source input\n"
        executable = b"sealed executable bytes"
        metallib = b"sealed shader bytes"
        source_hash = _digest(source)
        executable_hash = _digest(executable)
        metallib_hash = _digest(metallib)
        build = {
            "schema": "numi.cardboard.source-build.v1",
            "success": True,
            "source_snapshot": "/sealed/source",
            "build_dir": "/sealed/build",
            "source_sha256": {"matter/input.hpp": source_hash},
            "artifacts_sha256": {
                "/sealed/build/probe": executable_hash,
                "/sealed/build/shaders/Matter.metallib": metallib_hash,
            },
        }
        build_bytes = _write_json(case / "inputs/external/source-build.json", build)
        _write_json(case / "inputs/source/matter/input.hpp", {"unused": True})
        (case / "inputs/source/matter/input.hpp").write_bytes(source)
        (case / "inputs/runtime/probe").parent.mkdir(parents=True, exist_ok=True)
        (case / "inputs/runtime/probe").write_bytes(executable)
        (case / "inputs/runtime/shaders/Matter.metallib").parent.mkdir(parents=True, exist_ok=True)
        (case / "inputs/runtime/shaders/Matter.metallib").write_bytes(metallib)

        output = case / "output"
        _write_json(output / "manifest.json", {
            "schema": "numi.cardboard.explicit-strip.v1",
            "solver": {"steps": 4, "hold_steps": 1, "unload_steps": 2,
                       "relax_steps": 1, "release_steps": 0},
        })
        _write_json(output / "result.json", {
            "schema": compare.RESULT_SCHEMA,
            "status": "failed",
            "accepted_steps": 1,
            "physical_validation": False,
            "failure": "native step rejected",
        })
        (output / "observations.csv").write_bytes(b"step,status\n0,accepted\n1,rejected\n")
        (output / "rejected-conditioning/AP.bin").parent.mkdir(parents=True, exist_ok=True)
        (output / "rejected-conditioning/AP.bin").write_bytes(bytes(range(128)))

        argv = ["/sealed/build/probe", "--preset", "source-grade",
                "--metallib", "/sealed/build/shaders/Matter.metallib",
                "--output", str(output)]
        env = {"NM_FEM_RESIDUAL_ARMIJO_ROOT": "all"}
        if enabled:
            env.update({
                "NM_FEM_NEWTON_TRACE_ROOT": "1",
                "NM_FEM_OPERATOR_FD_SELECTOR": "1:0",
                "NM_FEM_OPERATOR_FD_CAPTURE_PATH": str(case / "capture.jsonl"),
            })
            (case / "capture.jsonl").write_bytes(b'{"schema":"test.capture"}\n')
        prediction = {
            "schema": compare.PREDICTION_SCHEMA,
            "executed_build_verified": True,
            "source_snapshot": "/sealed/source",
            "runtime_dir": "/sealed/build",
            "build_provenance": "/sealed/build/source-build.json",
            "external_inputs": ["/sealed/build/source-build.json"],
            "inputs_sha256": {
                "external/source-build.json": _digest(build_bytes),
                "source/matter/input.hpp": source_hash,
                "runtime/probe": executable_hash,
                "runtime/shaders/Matter.metallib": metallib_hash,
            },
            "source_binding_limitation": "test fixture",
            "argv": argv,
            "environment": env,
            "timeout_seconds": 60,
        }
        _write_json(case / "prediction.json", prediction)
        _write_json(case / "receipt.json", {
            "schema": compare.RECEIPT_SCHEMA,
            "execution_started": True,
            "source_inputs_unchanged": True,
            "changed_inputs": [],
            "exit_code": 2,
            "timed_out": False,
            "wall_seconds": 1.0,
        })

    def _prediction(self, case: Path):
        path = case / "prediction.json"
        value = json.loads(path.read_text())
        return path, value

    def test_identical_failed_attempts_report_parity_without_qualification(self):
        report = compare.compare_attempts(self.left, self.right)
        self.assertEqual(report["comparison"], "byte_identical")
        self.assertEqual(report["output_file_count"], 4)
        self.assertEqual(report["identical_file_count"], 4)
        self.assertEqual(report["attempts"]["diagnostic_enabled"]["verdict"], "failed_attempt")
        self.assertEqual(report["attempts"]["control"]["verdict"], "failed_attempt")
        self.assertEqual(report["attempts"]["diagnostic_enabled"]["expected_protocol_steps"], 8)
        self.assertFalse(report["qualification"].get("qualified", False))
        self.assertEqual(report["qualification"]["status"], "not_qualified")

    def test_physical_argv_difference_is_rejected(self):
        path, value = self._prediction(self.right)
        value["argv"].insert(-2, "--bend-angle")
        value["argv"].insert(-2, "1.0")
        _write_json(path, value)
        with self.assertRaisesRegex(compare.ComparisonError, "physical argv differs"):
            compare.compare_attempts(self.left, self.right)

    def test_unallowlisted_environment_difference_is_rejected(self):
        path, value = self._prediction(self.right)
        value["environment"]["NM_FEM_RESIDUAL_TOLERANCE"] = "looser"
        _write_json(path, value)
        with self.assertRaisesRegex(compare.ComparisonError, "environment changes are not allowlisted"):
            compare.compare_attempts(self.left, self.right)

    def test_input_archive_tampering_is_rejected_before_comparison(self):
        (self.right / "inputs/runtime/probe").write_bytes(b"modified executable")
        with self.assertRaisesRegex(compare.ComparisonError, "sealed input hash mismatch"):
            compare.compare_attempts(self.left, self.right)

    def test_different_sealed_executable_build_is_rejected(self):
        path, prediction = self._prediction(self.right)
        build_path = self.right / "inputs/external/source-build.json"
        build = json.loads(build_path.read_text())
        build["artifacts_sha256"]["/sealed/build/probe"] = _digest(b"other executable")
        build_bytes = _write_json(build_path, build)
        prediction["inputs_sha256"]["external/source-build.json"] = _digest(build_bytes)
        _write_json(path, prediction)
        with self.assertRaisesRegex(compare.ComparisonError, "executed binary archive does not match"):
            compare.compare_attempts(self.left, self.right)

    def test_output_difference_in_nested_rejected_arena_is_reported(self):
        (self.right / "output/rejected-conditioning/AP.bin").write_bytes(bytes(range(127)))
        report = compare.compare_attempts(self.left, self.right)
        self.assertEqual(report["comparison"], "different")
        self.assertEqual(report["differing_files"], ["rejected-conditioning/AP.bin"])
        self.assertEqual(report["attempts"]["control"]["verdict"], "failed_attempt")

    def test_timeout_or_input_mutation_receipt_is_rejected(self):
        receipt_path = self.right / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["timed_out"] = True
        _write_json(receipt_path, receipt)
        with self.assertRaisesRegex(compare.ComparisonError, "timed out/changed inputs"):
            compare.compare_attempts(self.left, self.right)

    def test_receipt_detecting_input_mutation_is_rejected(self):
        receipt_path = self.right / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["source_inputs_unchanged"] = False
        receipt["changed_inputs"] = ["/sealed/source/matter/input.hpp"]
        _write_json(receipt_path, receipt)
        with self.assertRaisesRegex(compare.ComparisonError, "timed out/changed inputs"):
            compare.compare_attempts(self.left, self.right)

    def test_capture_path_cannot_escape_attempt_directory(self):
        path, value = self._prediction(self.left)
        value["environment"]["NM_FEM_OPERATOR_FD_CAPTURE_PATH"] = str(self.root / "outside.jsonl")
        _write_json(path, value)
        with self.assertRaisesRegex(compare.ComparisonError, "inside its own attempt directory"):
            compare.compare_attempts(self.left, self.right)

    def test_exclusive_report_writer_does_not_overwrite(self):
        destination = self.root / "report.json"
        destination.write_text("preserve me")
        with self.assertRaises(FileExistsError):
            compare.write_report_exclusive(destination, {"ok": True})
        self.assertEqual(destination.read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
