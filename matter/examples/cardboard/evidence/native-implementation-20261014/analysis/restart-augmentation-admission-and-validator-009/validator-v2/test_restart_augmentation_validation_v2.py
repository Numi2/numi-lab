#!/usr/bin/env python3
"""Bounded source-capture checks for restart augmentation v2."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path("/Users/home/cardboard-implementation-20261014")
PACKAGE = ROOT / "analysis/restart-augmentation-admission-and-validator-009"
V2 = PACKAGE / "validator-v2/validate_restart_augmentation.py"
SOURCE323 = ROOT / "solver/two-env-current-newton-augmentation-off-bounded-323"
SOURCE182 = ROOT / "solver/selected-newton-regional-rap-and-operator-capture-182"
SOURCE337 = ROOT / "solver/current-newton-augmentation-on-N16-337"

SPEC = importlib.util.spec_from_file_location("restart_augmentation_validation_v2", V2)
assert SPEC is not None and SPEC.loader is not None
v2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v2
SPEC.loader.exec_module(v2)

BASE_TEST_SPEC = importlib.util.spec_from_file_location(
    "restart_augmentation_validation_base_tests",
    PACKAGE / "test_restart_augmentation_validation.py")
assert BASE_TEST_SPEC is not None and BASE_TEST_SPEC.loader is not None
base_tests = importlib.util.module_from_spec(BASE_TEST_SPEC)
sys.modules[BASE_TEST_SPEC.name] = base_tests
BASE_TEST_SPEC.loader.exec_module(base_tests)
base_tests.validator = v2


class RestartAugmentationV2Tests(unittest.TestCase):
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
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "base_operator_consistent_diagnostic")
            aug = report["restart_augmentation_validation"]
            self.assertEqual(aug["status"], "no_augmentation_attempt_observed")
            self.assertEqual(aug["attempted_environment_count"], 0)
            self.assertEqual([e["attempt"] for e in aug["environments"]], [False, False])
            self.assertEqual(aug["environments"][0]["non_attempt_factor_metadata"]["factor_valid_word"], 1)
            self.assertEqual(aug["environments"][0]["unattempted_augmented_factor_scratch"]["max_abs_matrix_word"], 0.0)
            self.assertEqual(report["run"]["status"], "failed")
            self.assertEqual(report["run"]["accepted_steps"], 0)
            self.assertFalse(report["run"]["physical_validation"])

    def test_augmented_per_environment_synthetic_contract(self):
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
            self.assertTrue(report["base_capture_integrity"]["basis_orthogonality_passed_all_attempted_environments"])
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
