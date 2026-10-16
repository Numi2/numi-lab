import hashlib
import json
import errno
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

import cardboard_diagnostic_history_compare as compare
import cardboard_source_bound_run_support as runner_support


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

    def _configure_regional_rap_capture(self, case: Path):
        prediction_path, prediction = self._prediction(case)
        capture_base = case / "rap-only-capture"
        prediction["environment"].update({
            "NM_FEM_OPERATOR_FD_SELECTOR": "1:0",
            "NM_FEM_OPERATOR_FD_CAPTURE_PATH": str(capture_base),
            "NM_FEM_REGIONAL_RAP_CAPTURE_ONLY": "1",
        })
        _write_json(prediction_path, prediction)
        (case / "capture.jsonl").unlink(missing_ok=True)

        names = {
            "candidate_nodes": (b"nodes", "uint8"),
            "material_history": (b"history", "uint8"),
            "scheduler_exponents": (b"sched", "uint8"),
            "deformation_high": (b"fhigh", "uint8"),
            "deformation_low": (b"flow", "uint8"),
            "P": (b"P", "uint8"),
            "effective_P": (b"EP", "uint8"),
            "AP": (b"AP", "uint8"),
            "pivoted_RAP": (b"RAP", "uint8"),
            "pivots": (b"piv", "uint8"),
            "factor_valid": (b"valid", "uint8"),
        }
        payload = bytearray()
        arrays = []
        spans = {}
        for name, (data, dtype) in names.items():
            offset = len(payload)
            payload.extend(data)
            spans[name] = (offset, len(data))
            arrays.append({
                "name": name,
                "dtype": dtype,
                "shape": [len(data)],
                "offset": offset,
                "byte_count": len(data),
                "element_bytes": 1,
                "sha256": _digest(data),
            })
        payload_path = Path(str(capture_base) + ".regional.bin")
        payload_path.write_bytes(payload)
        identity = hashlib.sha256()
        for name in ("candidate_nodes", "material_history", "scheduler_exponents",
                     "deformation_high", "deformation_low"):
            offset, size = spans[name]
            identity.update(payload[offset:offset + size])
        node_offset, node_size = spans["candidate_nodes"]
        manifest = {
            "schema": "numi.matter.regional_rap_capture.v1",
            "byte_order": "little_endian",
            "layout": (
                "C-contiguous; environment-major; each array is tightly concatenated with no padding; "
                "float32x4 elements store four adjacent little-endian lanes"),
            "capture_point": "after_regional_rap_factor_barrier_before_fgmres_correction",
            "source_physics_fingerprint": "0123456789abcdef",
            "device_program_fingerprint": "fedcba9876543210",
            "root": 1,
            "iteration": 0,
            "environment_count": 1,
            "node_count": 1,
            "tetrahedron_count": 1,
            "mode_count": 1,
            "preconditioner_selector": 2,
            "graph_schwarz": {"enabled": False},
            "material_state_stride": 16,
            "substep": 0,
            "microtick": 0,
            "microtick_count": 1,
            "linearization_generation": 0,
            "dt_seconds": 1e-4,
            "tangent_cache_used": False,
            "candidate_linearization_sha256": [{
                "environment": 0,
                "sha256": identity.hexdigest(),
                "candidate_nodes_sha256": _digest(payload[node_offset:node_offset + node_size]),
            }],
            "payload": {
                "file": str(payload_path),
                "byte_count": len(payload),
                "sha256": _digest(bytes(payload)),
            },
            "arrays": arrays,
        }
        _write_json(Path(str(capture_base) + ".regional.json"), manifest)
        return capture_base

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

    def test_capture_without_newton_trace_retains_same_parity_contract(self):
        path, value = self._prediction(self.left)
        del value["environment"]["NM_FEM_NEWTON_TRACE_ROOT"]
        _write_json(path, value)
        report = compare.compare_attempts(self.left, self.right)
        self.assertEqual(report["comparison"], "byte_identical")
        self.assertEqual(report["qualification"]["status"], "not_qualified")
        value["environment"]["NM_FEM_NEWTON_TRACE_ROOT"] = "0"
        _write_json(path, value)
        with self.assertRaisesRegex(compare.ComparisonError, "diagnostic-enabled"):
            compare.compare_attempts(self.left, self.right)

    def test_output_pair_may_precede_bound_material_arguments(self):
        for case in (self.left, self.right):
            path, value = self._prediction(case)
            value["argv"].extend(["--liner-material", "/sealed/source/liner.nmatter"])
            _write_json(path, value)
        report = compare.compare_attempts(self.left, self.right)
        self.assertEqual(report["comparison"], "byte_identical")
        path, value = self._prediction(self.right)
        value["argv"][-1] = "/sealed/source/other.nmatter"
        _write_json(path, value)
        with self.assertRaisesRegex(compare.ComparisonError, "physical argv differs"):
            compare.compare_attempts(self.left, self.right)

    def test_late_control_linear_capture_is_observational_only(self):
        path, value = self._prediction(self.left)
        value["environment"].update({"NM_FEM_OPERATOR_FD_SELECTOR": "3:23",
                                     "NM_FEM_NEWTON_TRACE_ROOT": "3",
                                     "NM_FEM_TRUE_RESIDUAL_ROOT": "3",
                                     "NM_FEM_TRUE_RESIDUAL_RESTART": "0",
                                     "NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT": "3"})
        _write_json(path, value)
        report = compare.compare_attempts(self.left, self.right)
        self.assertEqual(report["comparison"], "byte_identical")
        value["environment"]["NM_FEM_TRUE_RESIDUAL_RESTART"] = "1"
        _write_json(path, value)
        with self.assertRaisesRegex(compare.ComparisonError, "replacement changes the solver"):
            compare.compare_attempts(self.left, self.right)
        value["environment"]["NM_FEM_TRUE_RESIDUAL_RESTART"] = "0"
        value["environment"]["NM_FEM_TRUE_RESIDUAL_ROOT"] = "4"
        _write_json(path, value)
        with self.assertRaisesRegex(compare.ComparisonError, "captured control step"):
            compare.compare_attempts(self.left, self.right)

    def test_private_epsilon_scale_is_allowlisted_without_qualification(self):
        path, value = self._prediction(self.left)
        value["environment"]["NM_FEM_OPERATOR_FD_EPSILON_SCALE"] = ".125"
        _write_json(path, value)
        report = compare.compare_attempts(self.left, self.right)
        self.assertEqual(report["comparison"], "byte_identical")
        self.assertEqual(report["qualification"]["status"], "not_qualified")

    def test_selected_true_residual_newton_must_match_capture(self):
        path, value = self._prediction(self.left)
        value["environment"].update({"NM_FEM_OPERATOR_FD_SELECTOR": "3:23",
                                     "NM_FEM_NEWTON_TRACE_ROOT": "3",
                                     "NM_FEM_TRUE_RESIDUAL_ROOT": "3",
                                     "NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR": "23"})
        _write_json(path, value)
        self.assertEqual(compare.compare_attempts(self.left, self.right)["comparison"],
                         "byte_identical")
        for invalid in ("24", "257", "-1", "1.0"):
            value["environment"]["NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR"] = invalid
            _write_json(path, value)
            with self.assertRaises(compare.ComparisonError):
                compare.compare_attempts(self.left, self.right)
        value["environment"]["NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR"] = "23"
        del value["environment"]["NM_FEM_TRUE_RESIDUAL_ROOT"]
        _write_json(path, value)
        with self.assertRaisesRegex(compare.ComparisonError, "invalid true residual Newton"):
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

    def test_regional_rap_only_sidecars_are_integrity_checked(self):
        self._configure_regional_rap_capture(self.left)
        report = compare.compare_attempts(self.left, self.right)
        self.assertEqual(report["comparison"], "byte_identical")
        self.assertEqual(report["qualification"]["status"], "not_qualified")

    def test_regional_rap_only_requires_selector_and_capture_path(self):
        environment = {"NM_FEM_REGIONAL_RAP_CAPTURE_ONLY": "1"}
        with self.assertRaisesRegex(compare.ComparisonError, "requires both"):
            compare._validate_diagnostic_capture(self.left, environment)
        with self.assertRaisesRegex(compare.ComparisonError, "requires both"):
            compare._validate_diagnostic_capture(self.left, {
                "NM_FEM_REGIONAL_RAP_CAPTURE_ONLY": "1",
                "NM_FEM_OPERATOR_FD_SELECTOR": "1:0",
            })
        with self.assertRaisesRegex(compare.ComparisonError, "exactly 0 or 1"):
            compare._validate_diagnostic_capture(self.left, {
                "NM_FEM_REGIONAL_RAP_CAPTURE_ONLY": "true",
            })

    def test_regional_rap_only_rejects_missing_or_mismatched_sidecars(self):
        base = self._configure_regional_rap_capture(self.left)
        manifest_path = Path(str(base) + ".regional.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["root"] = 2
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(compare.ComparisonError, "root does not match"):
            compare.compare_attempts(self.left, self.right)
        manifest["root"] = 1
        _write_json(manifest_path, manifest)
        payload_path = Path(str(base) + ".regional.bin")
        payload_path.unlink()
        with self.assertRaisesRegex(compare.ComparisonError, "payload is missing"):
            compare.compare_attempts(self.left, self.right)

    def test_regional_rap_only_rejects_payload_and_array_hash_mismatch(self):
        base = self._configure_regional_rap_capture(self.left)
        payload_path = Path(str(base) + ".regional.bin")
        payload_path.write_bytes(payload_path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(compare.ComparisonError, "payload size or SHA-256"):
            compare.compare_attempts(self.left, self.right)
        base = self._configure_regional_rap_capture(self.left)
        manifest_path = Path(str(base) + ".regional.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["arrays"][0]["sha256"] = "0" * 64
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(compare.ComparisonError, "array hash mismatch"):
            compare.compare_attempts(self.left, self.right)

    def test_exclusive_report_writer_does_not_overwrite(self):
        destination = self.root / "report.json"
        destination.write_text("preserve me")
        with self.assertRaises(FileExistsError):
            compare.write_report_exclusive(destination, {"ok": True})
        self.assertEqual(destination.read_text(), "preserve me")


class SourceBoundTimeoutReceiptTests(unittest.TestCase):
    class _NeverExits:
        pid = 9123

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("sealed-probe", timeout)

    def test_sigkill_failure_keeps_timeout_and_original_os_error(self):
        state = {"timed_out": False, "termination_errors": []}
        signals = []

        def kill_group(_pid, signum):
            signals.append(signum)
            if signum == signal.SIGKILL:
                raise PermissionError(errno.EPERM, "operation not permitted")

        with self.assertRaises(PermissionError) as raised:
            runner_support.wait_with_timeout(
                self._NeverExits(), 1, state=state,
                kill_process_group=kill_group, term_grace_seconds=0.01)
        with tempfile.TemporaryDirectory() as temp:
            receipt_path = Path(temp) / "timeout-error-receipt.json"
            runner_support.write_failure_receipt(
                receipt_path, raised.exception, state=state,
                execution_started=True, wall_seconds=25.0,
                finished_at_utc="2026-10-15T00:00:00+00:00")
            receipt = json.loads(receipt_path.read_text())
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(receipt["schema"], compare.RECEIPT_SCHEMA)
        self.assertTrue(receipt["timed_out"])
        self.assertEqual(receipt["error_type"], "PermissionError")
        self.assertIn("operation not permitted", receipt["reason"])
        self.assertEqual(len(receipt["termination_errors"]), 1)
        self.assertEqual(receipt["termination_errors"][0]["signal_name"], "SIGKILL")
        self.assertEqual(receipt["termination_errors"][0]["errno"], errno.EPERM)
        self.assertIsNone(receipt["source_inputs_unchanged"])
        self.assertIsNone(receipt["changed_inputs"])

    def test_sigterm_error_is_retained_if_sigkill_completes(self):
        state = {"timed_out": False, "termination_errors": []}
        signals = []

        class ExitsAfterKill:
            pid = 9124
            waits = 0

            def wait(self, timeout=None):
                self.waits += 1
                if self.waits <= 2:
                    raise subprocess.TimeoutExpired("sealed-probe", timeout)
                return -signal.SIGKILL

        def kill_group(_pid, signum):
            signals.append(signum)
            if signum == signal.SIGTERM:
                raise ProcessLookupError(errno.ESRCH, "process group disappeared")

        code = runner_support.wait_with_timeout(
            ExitsAfterKill(), 1, state=state,
            kill_process_group=kill_group, term_grace_seconds=0.01)
        self.assertEqual(code, -signal.SIGKILL)
        self.assertTrue(state["timed_out"])
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(state["termination_errors"][0]["signal_name"], "SIGTERM")


if __name__ == "__main__":
    unittest.main()
