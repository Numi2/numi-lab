#!/usr/bin/env python3
"""Bounded source-capture checks for restart augmentation v2."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/Users/home/cardboard-implementation-20261014")
PACKAGE = ROOT / "analysis/restart-augmentation-admission-and-validator-009"
V2 = PACKAGE / "validator-v4-a-conjugacy/validate_restart_augmentation.py"
SOURCE323 = ROOT / "solver/two-env-current-newton-augmentation-off-bounded-323"
SOURCE182 = ROOT / "solver/selected-newton-regional-rap-and-operator-capture-182"
SOURCE337 = ROOT / "solver/current-newton-augmentation-on-N16-337"
SOURCE341 = ROOT / "solver/current-newton-augmentation-on-N28-capture27-341"
SOURCE344 = ROOT / "solver/reset-order-augmentation-on-N28-capture27-344"

SPEC = importlib.util.spec_from_file_location("restart_augmentation_validation_v4", V2)
assert SPEC is not None and SPEC.loader is not None
v2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v2
SPEC.loader.exec_module(v2)

BASE_TEST_SPEC = importlib.util.spec_from_file_location(
    "restart_augmentation_validation_base_tests_v4",
    PACKAGE / "validator-v4-a-conjugacy/test_restart_augmentation_validation.py")
assert BASE_TEST_SPEC is not None and BASE_TEST_SPEC.loader is not None
base_tests = importlib.util.module_from_spec(BASE_TEST_SPEC)
sys.modules[BASE_TEST_SPEC.name] = base_tests
BASE_TEST_SPEC.loader.exec_module(base_tests)
base_tests.validator = v2


def set_target_scales_to_captured_base_rap_medians(operator_path: Path,
                                                   conditioning_path: Path) -> np.ndarray:
    """Update only the synthetic per-environment target metadata and hash."""
    capture = v2.load_rich_capture(operator_path)
    conditioning = v2.load_conditioning(conditioning_path)
    envs = int(capture.header["environment_count"])
    nodes = int(capture.header["node_count_per_environment"])
    modes = int(conditioning.manifest["mode_count"])
    ep = v2._array(conditioning, "effective_P", "<f4", (envs, modes, nodes, 4))
    ap = v2._array(conditioning, "AP", "<f4", (envs, modes, nodes, 4))
    assembler = v2.load_operator_assembler(v2.DEFAULT_ASSEMBLER)
    targets = np.zeros(envs, dtype="<f4")
    for env in range(envs):
        _, _, _, _, meta = assembler.validate_and_assemble(
            capture.header, capture.nodes, capture.tets, capture.integrity, env)
        free = meta["free_nodes"]
        pmat = ep[env][:, free, :3].astype(np.float64).transpose(1, 2, 0).reshape(-1, modes)
        apmat = ap[env][:, free, :3].astype(np.float64).transpose(1, 2, 0).reshape(-1, modes)
        rap = pmat.T @ apmat
        diagonal = np.abs(np.diag(rap))
        if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.0):
            targets[env] = 0.0
        else:
            targets[env] = np.float32(np.partition(diagonal, modes // 2)[modes // 2])
    target_path = conditioning_path.parent / "restart_target_scales_by_environment.bin"
    target_path.write_bytes(targets.tobytes())
    manifest = json.loads(conditioning_path.read_text())
    import hashlib
    for desc in manifest["files"]:
        if desc["name"] == target_path.name:
            raw = target_path.read_bytes()
            desc["sha256"] = hashlib.sha256(raw).hexdigest()
            desc["bytes"] = len(raw)
            break
    conditioning_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return targets


class RestartAugmentationV3Tests(unittest.TestCase):
    def test_actual323_feature_off_fresh_base_ap(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            args = [sys.executable, str(V2), "--conditioning-only",
                    "--operator-capture", str(SOURCE323 / "operator-capture"),
                    "--conditioning-manifest", str(SOURCE323 / "output/rejected-conditioning-000001/manifest.json"),
                    "--run-result", str(SOURCE323 / "output/result.json"),
                    "--run-receipt", str(SOURCE323 / "receipt.json"),
                    "--run-manifest", str(SOURCE323 / "output/manifest.json"),
                    "--source-build", str(SOURCE323 / "inputs/external/source-build.json"),
                    "--output", str(output)]
            import subprocess
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "base_operator_consistent_diagnostic")
            self.assertEqual(report["linearization"]["root"], 0)
            self.assertEqual(report["linearization"]["newton_iteration"], 3)
            self.assertTrue(report["linearization"]["candidate_node_bytes_match_rich_center"])
            self.assertEqual(report["conditioning_generation"]["restart_diagnostics_available"], False)
            self.assertEqual(report["run"]["status"], "failed")
            self.assertEqual(report["run"]["accepted_steps"], 0)
            self.assertFalse(report["run"]["physical_validation"])
            self.assertTrue(report["source_build"]["success"])
            self.assertTrue(report["run"]["manifest"]["sha256"])
            self.assertLessEqual(max(e["base_AP_max_relative_error"] for e in report["environments"]), 2e-3)
            self.assertEqual([e["base_AP_all_modes_passed"] for e in report["environments"]], [True, True])

    def test_capture182_keeps_ap_mismatch_and_reports_fresh_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            args = [sys.executable, str(V2), "--conditioning-only",
                    "--operator-capture", str(SOURCE182 / "operator-capture"),
                    "--regional-json", str(SOURCE182 / "operator-capture.regional.json"),
                    "--conditioning-manifest", str(SOURCE182 / "output/rejected-conditioning-000004/manifest.json"),
                    "--output", str(output)]
            import subprocess
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "base_operator_inconsistent_diagnostic")
            comparison = report["regional_sidecar_exact_comparison"]
            self.assertFalse(comparison["AP"]["exact_match"])
            env = report["environments"]
            self.assertGreater(env[0]["base_AP_max_relative_error"], 0.9)
            self.assertLessEqual(env[1]["base_AP_max_relative_error"], 2e-3)
            # The mismatch is diagnostic evidence, not overwritten with the rich value.
            self.assertIn("conditioning_arrays", report["bindings"])

    def test_actual337_factor_fallback_is_not_misread_as_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            args = [sys.executable, str(V2), "--conditioning-only",
                    "--operator-capture", str(SOURCE337 / "operator-capture"),
                    "--conditioning-manifest", str(SOURCE337 / "output/rejected-conditioning-000001/manifest.json"),
                    "--run-result", str(SOURCE337 / "output/result.json"),
                    "--run-receipt", str(SOURCE337 / "receipt.json"),
                    "--run-manifest", str(SOURCE337 / "output/manifest.json"),
                    "--source-build", str(SOURCE337 / "inputs/external/source-build.json"),
                    "--output", str(output)]
            import subprocess
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "augmentation_capture_inconsistent_diagnostic")
            aug = report["restart_augmentation_validation"]
            self.assertEqual(aug["status"], "no_augmentation_attempt_observed")
            self.assertEqual(aug["attempted_environment_count"], 0)
            self.assertEqual([e["attempt"] for e in aug["environments"]], [False, False])
            self.assertEqual(aug["environments"][0]["non_attempt_factor_metadata"]["factor_valid_word"], 1)
            self.assertFalse(aug["target_scale_all_environments_passed"])
            self.assertFalse(aug["environments"][0]["target_scale_upper_median_check"]["passed"])
            self.assertEqual(aug["environments"][0]["unattempted_augmented_factor_scratch"]["max_abs_matrix_word"], 0.0)
            self.assertEqual(report["run"]["status"], "failed")
            self.assertEqual(report["run"]["accepted_steps"], 0)
            self.assertFalse(report["run"]["physical_validation"])

    def test_actual341_formed_but_rejected_attempt_and_identity_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            args = [sys.executable, str(V2), "--conditioning-only", "--require-attempt",
                    "--operator-capture", str(SOURCE341 / "operator-capture"),
                    "--conditioning-manifest", str(SOURCE341 / "output/rejected-conditioning-000001/manifest.json"),
                    "--run-result", str(SOURCE341 / "output/result.json"),
                    "--run-receipt", str(SOURCE341 / "receipt.json"),
                    "--run-manifest", str(SOURCE341 / "output/manifest.json"),
                    "--source-build", str(SOURCE341 / "inputs/external/source-build.json"),
                    "--output", str(output)]
            import subprocess
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "augmentation_capture_inconsistent_diagnostic")
            aug = report["restart_augmentation_validation"]
            self.assertEqual(aug["status"], "augmentation_attempts_inconsistent_diagnostic")
            self.assertEqual(aug["attempted_environment_count"], 1)
            self.assertEqual(aug["admitted_environment_count"], 0)
            env0 = aug["environments"][0]
            self.assertEqual(env0["attempt_state"], "attempted_rejected")
            self.assertEqual(env0["generation_words"], [0, 0, 27, 0])
            self.assertEqual(env0["attempt_count"], 1)
            self.assertEqual(env0["mode_valid_word"], 0)
            self.assertEqual(env0["factor_valid_word"], 1)
            self.assertEqual(env0["target_scale"], 0.0)
            self.assertGreater(env0["target_scale_upper_median_check"]["expected_upper_median_abs_base_RAP_diagonal"], 0.0)
            self.assertFalse(env0["target_scale_upper_median_check"]["passed"])
            self.assertGreater(env0["raw_norm_squared_before_cleanup"], 0.0)
            self.assertEqual(env0["mode_scale"], 0.0)
            self.assertTrue(env0["q_Aq_cleanup_exact_zero"])
            self.assertTrue(env0["fallback_factor"]["RAP_LU_reconstruction_gate_passed"])
            self.assertNotIn("fresh_Aq", env0)
            self.assertEqual([e["attempt"] for e in aug["environments"]], [True, False])
            self.assertTrue(report["source_build"]["success"])
            self.assertEqual(report["run"]["accepted_steps"], 0)
            self.assertFalse(report["run"]["physical_validation"])

    def test_admitted_mode_requirement_rejects_341_fail_closed_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            args = [sys.executable, str(V2), "--conditioning-only", "--require-admitted-augmentation",
                    "--operator-capture", str(SOURCE341 / "operator-capture"),
                    "--conditioning-manifest", str(SOURCE341 / "output/rejected-conditioning-000001/manifest.json"),
                    "--output", str(output)]
            import subprocess
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("admitted augmentation mode was required", result.stderr)
            failure = json.loads(output.read_text())
            self.assertEqual(failure["status"], "validation_failed")

    def test_no_attempt_accepts_nonzero_current_base_target_scale(self):
        with tempfile.TemporaryDirectory() as temp:
            old_err = np.seterr(all="ignore")
            try:
                conditioning = Path(temp) / "conditioning"
                shutil.copytree(SOURCE337 / "output/rejected-conditioning-000001", conditioning)
                conditioning = conditioning / "manifest.json"
                operator = SOURCE337 / "operator-capture"
                expected_targets = set_target_scales_to_captured_base_rap_medians(operator, conditioning)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    report = v2.validate_conditioning_only(operator, conditioning,
                        v2.DEFAULT_ASSEMBLER, 2e-3)
            finally:
                np.seterr(**old_err)
            quiet = report["restart_augmentation_validation"]["environments"][1]
            self.assertFalse(quiet["attempt"])
            self.assertEqual(quiet["base_target_scale_metadata"], float(expected_targets[1]))
            self.assertGreater(quiet["base_target_scale_metadata"], 0.0)
            self.assertTrue(quiet["target_scale_upper_median_check"]["passed"])
            self.assertEqual(report["status"], "base_operator_consistent_diagnostic")

    def test_augmented_per_environment_synthetic_A_conjugate_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            old_err = np.seterr(all="ignore")
            try:
                operator, regional, conditioning = base_tests.synthetic_augmented_payload(Path(temp))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    report = v2.validate(operator, regional, conditioning, None,
                        v2.DEFAULT_ASSEMBLER, True, 2e-3, 2e-4)
            finally:
                np.seterr(**old_err)
            self.assertEqual(report["status"], "internally_checked_diagnostic_capture")
            self.assertTrue(report["base_capture_integrity"]["base_AP_passed_all_environments"])
            self.assertTrue(report["base_capture_integrity"]["fresh_Aq_passed_all_attempted_environments"])
            self.assertTrue(report["base_capture_integrity"]["q_scale_passed_all_attempted_environments"])
            self.assertTrue(report["base_capture_integrity"]["A_conjugacy_passed_all_attempted_environments"])
            self.assertTrue(report["base_capture_integrity"]["augmented_RAP_passed_all_attempted_environments"])
            self.assertTrue(report["environments"][0]["attempt"])
            self.assertFalse(report["environments"][1]["attempt"])
            self.assertEqual(report["environments"][1]["generation_words"], [v2.UINT32_MAX] * 4)
            self.assertEqual(
                report["environments"][1]["non_attempt_factor_metadata"]["factor_valid_word"],
                0,
            )

            # Native snapshot 335 demonstrates that an environment with no
            # attempted q can retain a valid fallback factor word. It must be
            # reported as fallback metadata, never interpreted as q validity.
            factor_path = conditioning.parent / "restart_factor_valid_by_environment.bin"
            factor_values = np.fromfile(factor_path, dtype="<u4")
            factor_values[1] = 1
            factor_path.write_bytes(factor_values.tobytes())
            manifest = json.loads(conditioning.read_text())
            import hashlib
            for desc in manifest["files"]:
                if desc["name"] == factor_path.name:
                    raw = factor_path.read_bytes()
                    desc["sha256"] = hashlib.sha256(raw).hexdigest()
                    desc["bytes"] = len(raw)
                    break
            conditioning.write_text(json.dumps(manifest, indent=2) + "\n")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fallback_report = v2.validate(operator, regional, conditioning, None,
                    v2.DEFAULT_ASSEMBLER, True, 2e-3, 2e-4)
            quiet = fallback_report["environments"][1]
            self.assertFalse(quiet["attempt"])
            self.assertEqual(quiet["non_attempt_factor_metadata"]["factor_valid_word"], 1)
            self.assertIn("never evidence", quiet["non_attempt_factor_metadata"]["interpretation"])

            quiet_q = conditioning.parent / "restart_q_by_environment.bin"
            values = np.fromfile(quiet_q, dtype="<f4")
            values[882 * 4] = 1.0
            quiet_q.write_bytes(values.tobytes())
            manifest = json.loads(conditioning.read_text())
            for desc in manifest["files"]:
                if desc["name"] == quiet_q.name:
                    raw = quiet_q.read_bytes()
                    desc["sha256"] = hashlib.sha256(raw).hexdigest()
                    desc["bytes"] = len(raw)
                    break
            conditioning.write_text(json.dumps(manifest, indent=2) + "\n")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with self.assertRaisesRegex(ValueError, "quiet no-attempt payload"):
                    v2.validate(operator, regional, conditioning, None,
                        v2.DEFAULT_ASSEMBLER, True, 2e-3, 2e-4)

    def test_nonsymmetric_A_projection_requires_A_conjugacy_not_euclidean_q(self):
        pmat = np.asarray([[1.0], [0.0], [0.0]], dtype=np.float32)
        q = np.asarray([-0.5, 1.0, 0.0], dtype=np.float32)
        # A is intentionally nonsymmetric. For this q, P^T A q is zero while
        # P^T q is nonzero, matching the restart producer's RAP projection.
        aq = np.asarray([[2.0, 1.0, 0.0],
                         [0.0, 3.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32) @ q
        metrics = v2._a_conjugacy_metrics(pmat, q, aq)
        self.assertAlmostEqual(
            metrics["captured_Aq"]["fp64_widened_captured_float32_inputs"]["relative_P_transpose_Aq"],
            0.0, places=12)
        self.assertGreater(
            metrics["euclidean_P_transpose_q_diagnostic_only"]["relative_P_transpose_q"],
            0.4)
        bad_aq = np.asarray([1.0, 3.0, 0.0], dtype=np.float32)
        bad = v2._a_conjugacy_metrics(pmat, np.asarray([0.0, 1.0, 0.0], dtype=np.float32), bad_aq)
        self.assertGreater(
            bad["captured_Aq"]["fp64_widened_captured_float32_inputs"]["relative_P_transpose_Aq"],
            v2.MAX_BASIS_RELATIVE_LIMIT)

    def test_actual344_admitted_A_conjugate_restart_readback(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            args = [sys.executable, str(V2), "--conditioning-only", "--require-admitted-augmentation",
                    "--operator-capture", str(SOURCE344 / "operator-capture"),
                    "--conditioning-manifest", str(SOURCE344 / "output/rejected-conditioning-000001/manifest.json"),
                    "--run-result", str(SOURCE344 / "output/result.json"),
                    "--run-receipt", str(SOURCE344 / "receipt.json"),
                    "--run-manifest", str(SOURCE344 / "output/manifest.json"),
                    "--source-build", str(SOURCE344 / "inputs/external/source-build.json"),
                    "--output", str(output)]
            import subprocess
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "base_operator_consistent_diagnostic")
            aug = report["restart_augmentation_validation"]
            self.assertEqual(aug["status"], "augmentation_attempts_consistent_diagnostic")
            self.assertEqual(aug["attempted_environment_count"], 1)
            self.assertEqual(aug["admitted_environment_count"], 1)
            env = aug["environments"][0]
            self.assertEqual(env["generation_words"], [0, 0, 27, 0])
            self.assertTrue(env["target_scale_upper_median_check"]["passed"])
            self.assertLess(env["A_conjugacy"]["captured_Aq"]["fp64_widened_captured_float32_inputs"]["relative_P_transpose_Aq"], 2e-3)
            self.assertGreater(env["A_conjugacy"]["euclidean_P_transpose_q_diagnostic_only"]["relative_P_transpose_q"], 0.01)
            self.assertTrue(all(env["mode_energy_scale_checks"].values()))
            for action in ("captured_Aq", "fresh_independent_Aq"):
                energy = env["A_conjugacy"][action]["q_transpose_Aq_energy"]
                self.assertTrue(energy["sign_resolved_by_source_floor"])
                self.assertLess(energy["target_scale_abs_energy_relative_error"], 2e-3)
            self.assertTrue(env["fresh_Aq_gate_passed"])
            self.assertTrue(env["base_AP_gate_passed"])
            self.assertTrue(env["augmented_RAP_gate_passed"])
            self.assertEqual(report["run"]["accepted_steps"], 0)
            self.assertFalse(report["run"]["physical_validation"])

    def test_threshold_overrides_cannot_be_weakened(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            args = [sys.executable, str(V2), "--conditioning-only",
                    "--operator-capture", str(SOURCE323 / "operator-capture"),
                    "--conditioning-manifest", str(SOURCE323 / "output/rejected-conditioning-000001/manifest.json"),
                    "--base-ap-relative-limit", "0.5", "--output", str(output)]
            import subprocess
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("no looser than", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
