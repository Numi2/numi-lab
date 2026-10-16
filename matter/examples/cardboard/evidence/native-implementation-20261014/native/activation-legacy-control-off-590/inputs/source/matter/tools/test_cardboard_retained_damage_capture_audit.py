#!/usr/bin/env python3
"""Failure-injection tests for the source-bound retained-damage capture auditor."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


TOOL = Path(__file__).with_name("cardboard_retained_damage_capture_audit.py")
SPEC = importlib.util.spec_from_file_location("retained_damage_capture_audit", TOOL)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(*, active: bool = False) -> dict[str, float]:
    value = {name: 0.0 for name in (
        "sbar11", "sbar22", "sbar33", "sbar23", "sbar13", "sbar12",
        *(f"overlay_plastic_history_{axis}" for axis in range(6)),
        *(f"overlay_densification_{axis}" for axis in range(3)),
    )}
    value["accumulated_multiplier"] = 0.01 if active else 0.0
    return value


def response(stress: list[float], projected: dict[str, float], *, native: bool) -> dict[str, Any]:
    row = {
        "stress_Pa": stress,
        "projected_state": projected,
        "projection_residual_inf": 0.0,
    }
    if native:
        row.update({"valid": True, "projection_valid": True,
                    "projected_state_low_f32": {key: 0.0 for key in projected}})
    return row


def capture_document(run: Path, *, name: str = "test", branch_crossing: bool = False,
                     tangent_failure: bool = False) -> dict[str, Any]:
    source_mat = run / "inputs/source/material.nmatter"
    source_csv = run / "inputs/external/data.csv"
    executable = run / "inputs/runtime/checker"
    metallib = run / "inputs/runtime/shaders/checker.metallib"
    for path, content in ((source_mat, b"material source fixture\n"),
                          (source_csv, b"x,y\n0,0\n1,1\n"),
                          (executable, b"native executable fixture\n"),
                          (metallib, b"native metallib fixture\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    accepted = state()
    center = state()
    plus = state(active=branch_crossing)
    minus = state(active=branch_crossing)
    parameters = {
        **{f"hill_{name}": 0.0 for name in ("F", "G", "H", "L", "M", "N")},
        "q_floor": 1.0 if branch_crossing else 0.5,
        "sigma_ref": 1.0,
        **{f"overlay_plastic_hardening_{axis}": 0.0 for axis in range(6)},
        **{f"overlay_densification_modulus_pa_axis{axis}": 0.0 for axis in range(3)},
    }
    f_center = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    direction = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Binary-exact increment keeps this synthetic pair's realized direction
    # exactly equal to the requested one after binary32 storage.
    epsilon = 0.015625
    f_plus = list(f_center)
    f_minus = list(f_center)
    f_plus[0] = audit.f32(f_plus[0] + epsilon)
    f_minus[0] = audit.f32(f_minus[0] - epsilon)
    plus_stress = [epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    minus_stress = [-epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fd = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    native_ad = [2.0 if tangent_failure else 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    oracle_ad = list(fd)

    def relative_error(a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 / (
            sum(y * y for y in b) ** 0.5)

    plus_native = response(plus_stress, plus, native=True)
    minus_native = response(minus_stress, minus, native=True)
    plus_oracle = response(plus_stress, plus, native=False)
    minus_oracle = response(minus_stress, minus, native=False)
    row = {
        "requested_epsilon": epsilon,
        "plus_F_high_f32": f_plus, "plus_F_low_f32": [0.0] * 9,
        "plus_F_sum_f64": f_plus,
        "minus_F_high_f32": f_minus, "minus_F_low_f32": [0.0] * 9,
        "minus_F_sum_f64": f_minus,
        "realized_direction_f64": direction,
        "requested_direction_f32": direction,
        "direction_realization_relative_l2": 0.0,
        "midpoint_offset_inf_f": 0.0,
        "native_pair_valid": True, "oracle_pair_valid": True,
        "native_plus": plus_native, "native_minus": minus_native,
        "fp64_plus": plus_oracle, "fp64_minus": minus_oracle,
        "native_fd_action_pa_per_s": fd, "fp64_fd_action_pa_per_s": fd,
        "native_ad_action_at_requested_direction_pa_per_s": native_ad,
        "fp64_ad_action_at_requested_direction_pa_per_s": oracle_ad,
        "native_fd_vs_fp64_fd_relative_l2": 0.0,
        "native_ad_vs_fp64_fd_relative_l2": relative_error(native_ad, fd),
        "fp64_ad_vs_fp64_fd_relative_l2": 0.0,
        "fp64_fd_step_check_gate_1e_3": True,
        "oracle_branch_preserved": not branch_crossing,
        "native_branch_matches_fp64": True,
    }
    capture = {
        "schema": "numi.cardboard.retained-damage-capture.v3",
        "evidence_class": "synthetic_instrument_only_not_calibration",
        "physical_fem_steps": 0, "physical_calibration": False,
        "compensated_deformation_perturbations_enabled": True,
        "generic_compensated_implicit_projection": True,
        "F_f32": f_center, "F_low_f32": [0.0] * 9,
        "F_sum_f64": f_center, "direction_f32": direction,
        "accepted_history_f32": accepted,
        "cooked_parameters_f32": parameters,
        "center_native": response([0.0] * 9, center, native=True),
        "center_fp64_source_reprojection": response([0.0] * 9, center, native=False),
        "native_vs_fp64_center": {
            "stress_relative_l2": 0.0,
            "tangent_relative_l2": 0.0,
        },
        "compensated_deformation_perturbation_sweep": [row],
        "source_material": {"name": name, "path": str(source_mat),
                            "sha256": sha256(source_mat)},
        "hardening_dataset": {"path": str(source_csv), "sha256": sha256(source_csv)},
        "native_artifacts": {
            "executable_sha256": sha256(executable),
            "metallib_path": str(metallib), "metallib_sha256": sha256(metallib),
        },
    }
    return capture


class RetainedDamageCaptureAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        source_mat = self.run / "inputs/source/material.nmatter"
        source_csv = self.run / "inputs/external/data.csv"
        executable = self.run / "inputs/runtime/checker"
        metallib = self.run / "inputs/runtime/shaders/checker.metallib"
        capture_dir = self.run / "capture"
        capture_dir.mkdir(parents=True)
        capture_path = capture_dir / "synthetic.tangent-capture.json"
        write_json(capture_path, capture_document(self.run))
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.prediction_path = self.run / "prediction.json"
        self.receipt_path = self.run / "receipt.json"
        write_json(self.prediction_path, {
            "schema": "numi.cardboard.native-prediction.v1",
            "source_snapshot": str(self.snapshot),
            "runtime_dir": str(executable.parent),
            "argv": [str(executable)], "executed_build_verified": True,
            "inputs_sha256": {
                "source/material.nmatter": sha256(source_mat),
                "external/data.csv": sha256(source_csv),
                "runtime/checker": sha256(executable),
                "runtime/shaders/checker.metallib": sha256(metallib),
            },
        })
        write_json(self.receipt_path, {
            "exit_code": 0, "source_inputs_unchanged": True, "timed_out": False,
        })
        self.output = self.root / "audit.json"

    def run_audit(self) -> tuple[dict[str, Any], int]:
        return audit.audit_run(self.run, self.output)

    def mutate_capture(self, mutation) -> None:
        path = next((self.run / "capture").glob("*.tangent-capture.json"))
        capture = json.loads(path.read_text())
        mutation(capture)
        write_json(path, capture)

    def test_passing_smooth_capture_exits_zero(self) -> None:
        report, code = self.run_audit()
        self.assertEqual(code, audit.EXIT_PASS)
        self.assertEqual(report["overall"]["status"], "strict_gate_pass")
        self.assertEqual(report["captures"][0]["strict_gate"]["passing_epsilons"], [0.015625])

    def test_source_hash_mismatch_exits_invalid_evidence(self) -> None:
        material = self.run / "inputs/source/material.nmatter"
        material.write_bytes(b"mutated material\n")
        report, code = self.run_audit()
        self.assertEqual(code, audit.EXIT_INVALID_EVIDENCE)
        self.assertFalse(report["source_rebinding"]["all_bindings_match"])

    def test_wrong_low_word_and_missing_low_word_are_rejected(self) -> None:
        def wrong_low(capture):
            row = capture["compensated_deformation_perturbation_sweep"][0]
            row["plus_F_low_f32"][0] = 0.125
        self.mutate_capture(wrong_low)
        _, code = self.run_audit()
        self.assertEqual(code, audit.EXIT_INVALID_EVIDENCE)

        def missing_low(capture):
            row = capture["compensated_deformation_perturbation_sweep"][0]
            row["minus_F_low_f32"].pop()
        self.mutate_capture(missing_low)
        report, code = self.run_audit()
        self.assertEqual(code, audit.EXIT_INVALID_EVIDENCE)
        self.assertTrue(report["capture_errors"])

    def test_direction_metric_mismatch_is_rejected(self) -> None:
        def corrupt_metric(capture):
            capture["compensated_deformation_perturbation_sweep"][0][
                "direction_realization_relative_l2"] = 0.25
        self.mutate_capture(corrupt_metric)
        report, code = self.run_audit()
        self.assertEqual(code, audit.EXIT_INVALID_EVIDENCE)
        self.assertIn("direction error metric mismatch", report["capture_errors"][0]["error"])

    def test_branch_crossing_is_inconclusive_not_a_smooth_pass_or_fail(self) -> None:
        self.mutate_capture(lambda capture: capture.__setitem__(
            "compensated_deformation_perturbation_sweep",
            capture_document(self.run, branch_crossing=True)[
                "compensated_deformation_perturbation_sweep"]))
        report, code = self.run_audit()
        self.assertEqual(code, audit.EXIT_INCONCLUSIVE)
        self.assertEqual(report["overall"]["status"], "strict_gate_inconclusive")
        row = report["captures"][0]["pair_rows"][0]
        self.assertEqual(row["row_status"], "inconclusive_direction_or_branch")

    def test_smooth_pair_tangent_failure_exits_one(self) -> None:
        self.mutate_capture(lambda capture: capture.__setitem__(
            "compensated_deformation_perturbation_sweep",
            capture_document(self.run, tangent_failure=True)[
                "compensated_deformation_perturbation_sweep"]))
        report, code = self.run_audit()
        self.assertEqual(code, audit.EXIT_STRICT_FAILURE)
        self.assertEqual(report["overall"]["status"], "strict_gate_failure")
        self.assertEqual(report["captures"][0]["strict_gate"]["status"],
                         "failure_smooth_requested_pair")

    def test_zero_response_scale_is_inconclusive_not_a_strict_failure(self) -> None:
        def erase_signal(capture):
            row = capture["compensated_deformation_perturbation_sweep"][0]
            for side in ("native_plus", "native_minus", "fp64_plus", "fp64_minus"):
                row[side]["stress_Pa"] = [0.0] * 9
            for key in ("native_fd_action_pa_per_s", "fp64_fd_action_pa_per_s",
                        "native_ad_action_at_requested_direction_pa_per_s",
                        "fp64_ad_action_at_requested_direction_pa_per_s"):
                row[key] = [0.0] * 9
            for key in ("native_fd_vs_fp64_fd_relative_l2",
                        "native_ad_vs_fp64_fd_relative_l2",
                        "fp64_ad_vs_fp64_fd_relative_l2"):
                row[key] = None
        self.mutate_capture(erase_signal)
        report, code = self.run_audit()
        self.assertEqual(code, audit.EXIT_INCONCLUSIVE)
        self.assertEqual(report["captures"][0]["pair_rows"][0]["row_status"],
                         "inconclusive_below_response_scale")

    def test_cli_exit_code_tracks_strict_gate(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(TOOL), str(self.run), "--output", str(self.output)],
            text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, audit.EXIT_PASS, completed.stderr)
        self.assertTrue(self.output.is_file())

        failing_output = self.root / "failure.json"
        self.mutate_capture(lambda capture: capture.__setitem__(
            "compensated_deformation_perturbation_sweep",
            capture_document(self.run, tangent_failure=True)[
                "compensated_deformation_perturbation_sweep"]))
        completed = subprocess.run(
            [sys.executable, str(TOOL), str(self.run), "--output", str(failing_output)],
            text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, audit.EXIT_STRICT_FAILURE, completed.stderr)

        inconclusive_output = self.root / "inconclusive.json"
        self.mutate_capture(lambda capture: capture.__setitem__(
            "compensated_deformation_perturbation_sweep",
            capture_document(self.run, branch_crossing=True)[
                "compensated_deformation_perturbation_sweep"]))
        completed = subprocess.run(
            [sys.executable, str(TOOL), str(self.run), "--output", str(inconclusive_output)],
            text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, audit.EXIT_INCONCLUSIVE, completed.stderr)


if __name__ == "__main__":
    unittest.main()
