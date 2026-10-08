#!/usr/bin/env python3
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("readiness_p15_scene", ROOT / "prepare_final_plan.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def receipt(path, digest):
    return {"schema": "numi.human.resting-anatomy-receipt.v1",
            "payload": {"path": str(path), "sha256": digest}}


class DynamicSceneInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.nha = self.root / "candidate.nhanatomy"
        self.nha.write_bytes(b"candidate anatomy")
        self.nha_sha = prepare.sha(self.nha)
        self.base = self.root / "source-receipt.json"
        self.native = self.root / "composed-receipt.json"
        payload = receipt(self.nha, self.nha_sha)
        self.base.write_text(json.dumps(payload), encoding="utf-8")
        self.native.write_text(json.dumps(payload), encoding="utf-8")
        self.config = self.root / "resting-reference-respiration-derived.json"
        self.config.write_text(json.dumps({"effective_area_m2": 0.0187, "note": "caller selected"}),
                               encoding="utf-8")
        self.config_sha = prepare.sha(self.config)
        self.selected = prepare.validate_selected_scene_inputs(
            self.nha, self.nha_sha, self.base, prepare.sha(self.base),
            self.native, prepare.sha(self.native), self.config, self.config_sha)

    def tearDown(self):
        self.tmp.cleanup()

    def test_selected_nha_both_receipts_and_nondefault_config_are_bound(self):
        self.assertEqual(self.selected["nha_sha256"], self.nha_sha)
        self.assertEqual(self.selected["base_receipt_sha256"], prepare.sha(self.base))
        self.assertEqual(self.selected["native_receipt_sha256"], prepare.sha(self.native))
        self.assertEqual(self.selected["respiration_config_path"], str(self.config.resolve()))
        self.assertEqual(self.selected["respiration_config_sha256"], self.config_sha)

    def test_rejects_wrong_config_hash_or_receipt_payload(self):
        with self.assertRaisesRegex(ValueError, "selected respiration configuration path/hash mismatch"):
            prepare.validate_selected_scene_inputs(
                self.nha, self.nha_sha, self.base, prepare.sha(self.base),
                self.native, prepare.sha(self.native), self.config, "0" * 64)
        self.native.write_text(json.dumps(receipt(self.nha, "0" * 64)), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not bind the exact selected NHA"):
            prepare.validate_selected_scene_inputs(
                self.nha, self.nha_sha, self.base, prepare.sha(self.base),
                self.native, prepare.sha(self.native), self.config, self.config_sha)

    def test_assembly_resolution_follows_selected_respiration_config_not_761(self):
        config = Path(self.selected["respiration_config_path"])
        argv = ["native", "--resting-scene", "/assets/circ.json", str(config)]
        assembly = {
            "owner_cli": {
                "resolved_native_argv": argv,
                "asset_sha256": {str(config): self.config_sha},
                "requested": {"respiration_config": {
                    "path": str(config), "sha256": self.config_sha}},
            },
            "respiration_config": {
                "path": str(config), "sha256": self.config_sha,
                "source": "caller-pinned path/hash override"},
            "source_sha256": {str(config): self.config_sha},
        }
        invocation = {"argv": argv, "asset_sha256": {str(config): self.config_sha}}
        self.assertEqual(prepare.respiration_asset_from_assembly(
            assembly, invocation, self.selected), {str(config): self.config_sha})
        bad = json.loads(json.dumps(assembly))
        bad["respiration_config"]["path"] = str(self.root / "wrong.json")
        with self.assertRaisesRegex(ValueError, "exact caller-selected"):
            prepare.respiration_asset_from_assembly(bad, invocation, self.selected)


class Reference931InputContextTests(unittest.TestCase):
    def test_compares_invocation_bound_config_and_contact_geometry_bytes(self):
        reference = prepare.read_json(
            prepare.TERMINAL_931_INVOCATION, "931 invocation fixture")
        selected_path = Path(prepare.TERMINAL_931_RESPIRATION_CONFIG).resolve()
        selected = {
            "respiration_config_path": str(selected_path),
            "respiration_config_sha256": prepare.TERMINAL_931_RESPIRATION_CONFIG_SHA,
        }
        unchanged = prepare.reference_931_input_context(reference, selected)
        self.assertTrue(unchanged["respiration_configuration_byte_identical_to_931"])
        self.assertTrue(unchanged["all_mechanical_configuration_bytes_identical_to_931"])
        self.assertTrue(unchanged["all_contact_driving_geometry_bytes_identical_to_931"])
        self.assertTrue(unchanged["runtime_scalar_settings_identical_to_931"])
        self.assertTrue(unchanged["all_compared_physical_inputs_identical_to_931"])

        with tempfile.TemporaryDirectory(dir=prepare.READINESS_ROOT,
                                         prefix=".v015-reference-input-test-") as temp:
            changed_config = Path(temp) / "respiration.json"
            changed_config.write_text('{"effective_area_m2":0.019}', encoding="utf-8")
            candidate = json.loads(json.dumps(reference))
            argv = candidate["argv"]
            index = argv.index("--resting-scene")
            argv[index + 2] = str(changed_config)
            digest = prepare.sha(changed_config)
            candidate["asset_sha256"][str(changed_config)] = digest
            selected = {"respiration_config_path": str(changed_config.resolve()),
                        "respiration_config_sha256": digest}
            context = prepare.reference_931_input_context(candidate, selected)
            self.assertFalse(context["respiration_configuration_byte_identical_to_931"])
            self.assertFalse(context["all_mechanical_configuration_bytes_identical_to_931"])
            self.assertTrue(context["all_contact_driving_geometry_bytes_identical_to_931"])
            self.assertFalse(context["all_mechanical_configuration_bytes_identical_to_931"])
            self.assertIn("finite differences alone do not establish physiological acceptance",
                          context["interpretation"])

    def test_runtime_scalar_or_environment_changes_are_visible(self):
        reference = prepare.read_json(
            prepare.TERMINAL_931_INVOCATION, "931 invocation fixture")
        selected = {
            "respiration_config_path": str(prepare.TERMINAL_931_RESPIRATION_CONFIG.resolve()),
            "respiration_config_sha256": prepare.TERMINAL_931_RESPIRATION_CONFIG_SHA,
        }
        candidate = json.loads(json.dumps(reference))
        argv = candidate["argv"]
        argv[argv.index("--muscle-activation") + 1] = "0.011"
        context = prepare.reference_931_input_context(candidate, selected)
        self.assertTrue(context["all_mechanical_configuration_bytes_identical_to_931"])
        self.assertFalse(context["runtime_scalar_settings_identical_to_931"])
        self.assertFalse(context["all_compared_physical_inputs_identical_to_931"])
        self.assertFalse(context["runtime_scalar_settings"]["--muscle-activation"]["identical"])

    def test_rejects_unverified_candidate_contact_asset_hash(self):
        reference = prepare.read_json(
            prepare.TERMINAL_931_INVOCATION, "931 invocation fixture")
        selected = {
            "respiration_config_path": str(prepare.TERMINAL_931_RESPIRATION_CONFIG.resolve()),
            "respiration_config_sha256": prepare.TERMINAL_931_RESPIRATION_CONFIG_SHA,
        }
        candidate = json.loads(json.dumps(reference))
        contact_path = candidate["argv"][candidate["argv"].index("--skin-payload") + 1]
        candidate["asset_sha256"][contact_path] = "0" * 64
        with self.assertRaisesRegex(ValueError, "recorded hash is wrong"):
            prepare.reference_931_input_context(candidate, selected)


class MeasuredPreflightComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            dir=prepare.READINESS_ROOT, prefix=".v015-comparison-test-")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.scene = self.root / "native-run"
        self.scene.mkdir()
        self.trace = self.scene / "resting-coupled.csv"
        self.trace.write_text("fixture trace" + chr(10), encoding="utf-8")
        columns = {
            "field_%02d" % i: {"samples": 1250, "max_abs_delta": 0.001 * (i + 1),
                              "mean_signed_delta": -0.0001 * i}
            for i in range(60)
        }
        steps = list(range(8, 10001, 8))
        poses = {
            str(step): {"body_count": 86, "max_position_delta_m": 1e-4 * i,
                        "max_quaternion_component_delta": 2e-5 * i}
            for i, step in enumerate((0, 4991, 5375, 5759, 6111, 6495, 7743, 10000))
        }
        self.report = {
            "schema": "numi.human.final-20s-vs-931-comparison.v1",
            "status": "measured_no_parity_assumed",
            "qualification": "20 s regression only",
            "reference": str(prepare.E / "native-terminal-cycle-931"),
            "candidate": str(self.scene),
            "candidate_csv_sha256": prepare.sha(self.trace),
            "candidate_observations": 1250,
            "reference_roots": 10000,
            "terminal_log_proof": {
                "accepted_steps": 10000, "actual_simulated_s": 10000 * prepare.NATIVE_FLOAT_DT_S,
                "float32_dt_s": prepare.NATIVE_FLOAT_DT_S, "terminal_presentations": 1,
                "terminal_state_advanced_physics": False},
            "trace_cadence_comparison": {
                "reference_root_count": 10000, "reference_rows": 10000,
                "candidate_rows": 1250, "segment_steps": 8,
                "candidate_accepted_steps": steps,
                "all_values_exact": False, "columns": columns,
            },
            "captured_86_body_poses": poses,
            "invocation_contract": {"loaded_runtime_verified": True},
        }
        self.path = self.root / "comparison.json"
        self.write_report()

    def write_report(self):
        self.path.write_text(json.dumps(self.report), encoding="utf-8")
        self.digest = prepare.sha(self.path)

    def test_accepts_and_reports_real_parameter_dependent_differences(self):
        measured = prepare.validate_v015_preflight_comparison(
            self.path, self.digest, self.scene)
        self.assertFalse(measured["cadence_all_values_exact"])
        self.assertGreater(measured["column_differences"]["field_00"]["max_abs_delta"], 0)
        self.assertGreater(measured["captured_body_pose_differences"]["4991"]["max_position_delta_m"], 0)
        self.assertEqual(measured["terminal_log_proof"]["accepted_steps"], 10000)
        self.assertEqual(measured["scope"].split(";")[0], "Measured 20 s regression only")

    def test_rejects_wrong_csv_hash_or_accepted_endpoint(self):
        self.report["candidate_csv_sha256"] = "0" * 64
        self.write_report()
        with self.assertRaisesRegex(ValueError, "not bound to this accepted 20 s run"):
            prepare.validate_v015_preflight_comparison(self.path, self.digest, self.scene)
        self.report["candidate_csv_sha256"] = prepare.sha(self.trace)
        self.report["terminal_log_proof"]["accepted_steps"] = 9999
        self.write_report()
        with self.assertRaisesRegex(ValueError, "exact accepted terminal"):
            prepare.validate_v015_preflight_comparison(self.path, self.digest, self.scene)
        self.report["terminal_log_proof"]["accepted_steps"] = 10000
        self.report["trace_cadence_comparison"]["candidate_accepted_steps"][-1] = 9999
        self.write_report()
        with self.assertRaisesRegex(ValueError, "cadence contract"):
            prepare.validate_v015_preflight_comparison(self.path, self.digest, self.scene)

    def test_rejects_nonfinite_delta_and_missing_pose(self):
        self.report["trace_cadence_comparison"]["columns"]["field_00"]["max_abs_delta"] = math.nan
        self.write_report()
        with self.assertRaisesRegex(ValueError, "invalid numerical field differences"):
            prepare.validate_v015_preflight_comparison(self.path, self.digest, self.scene)
        self.report["trace_cadence_comparison"]["columns"]["field_00"]["max_abs_delta"] = 0.001
        self.report["captured_86_body_poses"].pop("10000")
        self.write_report()
        with self.assertRaisesRegex(ValueError, "finite accepted-state body pose comparisons"):
            prepare.validate_v015_preflight_comparison(self.path, self.digest, self.scene)


if __name__ == "__main__":
    unittest.main()
