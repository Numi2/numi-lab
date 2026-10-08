#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("prepare_final_plan.py")
spec = importlib.util.spec_from_file_location("readiness_prepare", str(MODULE_PATH))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class TerminalCapturePlanTests(unittest.TestCase):
    def setUp(self):
        self.schedule = prepare.accepted_geometry_capture_schedule()

    def test_readiness_owned_paths_follow_the_checked_out_package(self):
        local_root = Path(prepare.__file__).resolve().parent
        self.assertEqual(prepare.READINESS_ROOT, local_root)
        for path in (prepare.READINESS_SCRIPT, prepare.READINESS_ANALYZER,
                     prepare.READINESS_README, prepare.READINESS_REVISION,
                     prepare.CAPTURE_PLAN_TESTS, prepare.OWNER_LINEAGE_TESTS):
            self.assertEqual(path.parent, local_root)
        self.assertEqual(prepare.TREATMENT_PROBE_RECORDER,
                         local_root / "record_treatment_program_probe.py")
        self.assertEqual(prepare.TREATMENT_PROBE_TESTS,
                         local_root / "test_record_treatment_program_probe_v015.py")
        self.assertEqual(prepare.PROGRAM_PROBE_NAME, "treatment-program-probe-v015")

    def test_each_arm_has_seven_interior_steps_and_true_terminal(self):
        self.assertEqual(self.schedule["accepted_horizon_steps"], 155000)
        self.assertEqual(self.schedule["accepted_duration_s"], 310.0)
        self.assertEqual(self.schedule["terminal"]["accepted_step_id"], 155000)
        self.assertAlmostEqual(self.schedule["terminal"]["accepted_time_s"], 310.000014724, places=9)
        self.assertEqual(self.schedule["terminal"]["nominal_time_s"], 310.0)
        self.assertTrue(self.schedule["terminal"]["n_minus_one_is_not_terminal"])
        for arm in ("control", "treatment"):
            events = self.schedule["arms"][arm]["events"]
            self.assertEqual(len(events), 8)
            self.assertEqual(sum(event["terminal"] for event in events), 1)
            self.assertEqual(events[-1]["accepted_step_id"], 155000)
            self.assertAlmostEqual(events[-1]["accepted_time_s"], 310.000014724, places=9)
            self.assertEqual(events[-1]["nominal_time_s"], 310.0)
            self.assertNotIn(154999, self.schedule["arms"][arm]["step_ids"])
            for event in events[:-1]:
                step = event["accepted_step_id"]
                self.assertAlmostEqual(event["accepted_time_s"],
                                       step * prepare.NATIVE_FLOAT_DT_S, places=12)
                self.assertEqual(event["nominal_time_s"], step * 0.002)
                self.assertEqual((step + 1) % 32, 0)

    def test_exact_predeclared_ids_and_times(self):
        self.assertEqual(self.schedule["arms"]["control"]["step_ids"],
                         [47519, 49151, 51903, 54047, 55647, 152191, 154143, 155000])
        self.assertEqual(self.schedule["arms"]["treatment"]["step_ids"],
                         [47519, 49151, 51903, 54047, 55647, 152447, 154367, 155000])
        self.assertEqual([round(x["accepted_time_s"], 3)
                          for x in self.schedule["arms"]["control"]["events"]],
                         [95.038, 98.302, 103.806, 108.094, 111.294,
                          304.382, 308.286, 310.0])
        self.assertEqual([round(x["accepted_time_s"], 3)
                          for x in self.schedule["arms"]["treatment"]["events"]],
                         [95.038, 98.302, 103.806, 108.094, 111.294,
                          304.894, 308.734, 310.0])

    def test_binding_updates_plan_templates_and_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = root / "preflight"
            out = root / "draft"
            scene.mkdir()
            out.mkdir()
            parent_path = scene / "invocation.json"
            prior_capture_value = "0,31,63,95,127,159,191,223"
            parent = {"argv": ["/native", "--muscle-step-count", "10000",
                                "--muscle-step-seconds", "0.002"],
                      "environment": {"PATH": "/usr/bin",
                                      prepare.CAPTURE_ENV_KEY: prior_capture_value},
                      "asset_sha256": {"/native": "a" * 64}}
            parent_path.write_text(json.dumps(parent))
            plan = {
                "design": {},
                "trials": [
                    {"id": "resting-baseline", "argv": ["python", "run", "--invocation", "old",
                                                         "--native-build-identity", "old-identity",
                                                         "--native-build-identity-sha256", "0" * 64]},
                    {"id": "resting-drive-half", "argv": ["python", "run", "--invocation", "old",
                                                          "--native-build-identity", "old-identity",
                                                          "--native-build-identity-sha256", "0" * 64]}],
                "artifacts": [], "instrument": {"artifacts": []}}
            base_identity = {"schema": "numi.human-resting.native-paired-build-identity.v1",
                             "native_invocation": {"path": str(parent_path.resolve()),
                                                   "sha256": prepare.sha(parent_path),
                                                   "asset_sha256": parent["asset_sha256"]},
                             "runtime_dependency_resolution": []}
            (out / "native-build-identity.json").write_text(json.dumps(base_identity))
            (out / "calibration.json").write_text(json.dumps({"bindings": {}}))
            runtime_records = [{"configured_path": "/build/dependency-" + str(i),
                                "resolved_path": "/frozen/dependency-" + str(i),
                                "asset_binding_path": "/frozen/dependency-" + str(i),
                                "sha256": format(i + 1, "064x"), "symlink_components": []}
                               for i in range(7)]
            with patch.object(prepare.owner, "native_scene_runtime_dependency_bindings",
                              return_value=runtime_records):
                result = prepare.bind_accepted_geometry_capture_plan(plan, out, scene, parent)
            schedule, template_paths, template_hashes, observed_parent, parent_sha = result
            self.assertEqual(observed_parent, parent_path.resolve())
            self.assertEqual(parent_sha, prepare.sha(parent_path))
            self.assertEqual(plan["design"]["accepted_geometry_capture_schedule"], schedule)
            for arm, trial_id in (("control", "resting-baseline"),
                                  ("treatment", "resting-drive-half")):
                trial = next(x for x in plan["trials"] if x["id"] == trial_id)
                self.assertEqual(trial["argv"][trial["argv"].index("--invocation") + 1],
                                 template_paths[arm])
                bound_identity_path = Path(trial["argv"][trial["argv"].index("--native-build-identity") + 1])
                bound_identity_sha = trial["argv"][trial["argv"].index("--native-build-identity-sha256") + 1]
                self.assertEqual(bound_identity_sha, prepare.sha(bound_identity_path))
                identity_bytes = bound_identity_path.read_bytes()
                bound_identity = json.loads(identity_bytes.decode("utf-8"))
                self.assertTrue(identity_bytes.endswith(b"\n"))
                self.assertEqual(bound_identity["native_invocation"]["path"], template_paths[arm])
                self.assertEqual(bound_identity["native_invocation"]["sha256"], template_hashes[arm])
                self.assertEqual(bound_identity["native_invocation"]["asset_sha256"], parent["asset_sha256"])
                self.assertEqual(bound_identity["runtime_dependency_resolution"], runtime_records)
                self.assertIn(str(bound_identity_path.resolve()), plan["artifacts"])
                template = json.loads(Path(template_paths[arm]).read_text())
                self.assertEqual(template["argv"], parent["argv"])
                self.assertEqual(template["environment"]["PATH"], "/usr/bin")
                self.assertEqual(template["environment"][prepare.CAPTURE_ENV_KEY],
                                 schedule["arms"][arm]["environment_value"])
                lineage = template["readiness_derived_capture_launch_template"]
                self.assertEqual(lineage["parent_invocation_sha256"], parent_sha)
                self.assertEqual(lineage["parent_capture_environment_value"], prior_capture_value)
                self.assertEqual(set(template["environment"]), set(parent["environment"]))
                for key, value in parent["environment"].items():
                    if key != prepare.CAPTURE_ENV_KEY:
                        self.assertEqual(template["environment"][key], value)
                self.assertEqual(prepare.sha(template_paths[arm]), template_hashes[arm])
                self.assertIn(template_paths[arm], plan["instrument"]["artifacts"])
                self.assertIn(str(bound_identity_path.resolve()), plan["instrument"]["artifacts"])
            calibration = json.loads((out / "calibration.json").read_text())
            schedule_path = str((out / prepare.CAPTURE_PLAN_FILE).resolve())
            self.assertEqual(calibration["bindings"][schedule_path], prepare.sha(schedule_path))
            for trial in plan["trials"]:
                identity_path = Path(trial["argv"][trial["argv"].index("--native-build-identity") + 1])
                self.assertEqual(calibration["bindings"][str(identity_path.resolve())], prepare.sha(identity_path))

    def test_terminal_receipt_must_be_N_at_310_seconds(self):
        receipt = {
            "accepted_step": 155000,
            "accepted_time_s": 155000 * prepare.NATIVE_FLOAT_DT_S,
            "physical_endpoint": "accepted", "surface_audit_endpoint": "passed",
            "accepted_root_fingerprint_hex": "0x1",
            "accepted_body_state_sha256": "a" * 64,
            "accepted_respiration_state_sha256": "b" * 64,
            "pack_file_sha256": "c" * 64,
        }
        result = prepare.validate_capture_receipt(self.schedule, "control", receipt)
        self.assertEqual(result, {"accepted_step": 155000,
                                  "accepted_time_s": 155000 * prepare.NATIVE_FLOAT_DT_S,
                                  "nominal_time_s": 310.0, "terminal": True})
        wrong_time = dict(receipt, accepted_time_s=309.998)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            prepare.validate_capture_receipt(self.schedule, "control", wrong_time)
        n_minus_one = dict(receipt, accepted_step=154999, accepted_time_s=310.0)
        with self.assertRaises(ValueError):
            prepare.validate_capture_receipt(self.schedule, "control", n_minus_one)

    def test_shared_json_writer_roundtrips_rewritten_native_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "native-build-identity.json"
            initial = {"schema": "test", "native_invocation": {"sha256": "a" * 64}}
            prepare.write_new_json(path, initial)
            updated = dict(initial, preparation_owner={"revision": "d81e236"})
            prepare.rewrite_json_document(path, updated)
            raw = path.read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            self.assertFalse(raw.endswith(b"\\n"))
            self.assertEqual(json.loads(raw.decode("utf-8")), updated)

    def test_only_capture_environment_delta_is_authorized(self):
        parent = {
            "argv": ["/bin/numi-human-native", "--input", "/tmp/scene"],
            "environment": {"PATH": "/usr/bin", "NUMI_HUMAN_SPLIT_STAND": "1"},
            "asset_sha256": {"/tmp/native": "a" * 64},
            "host": "Mini", "machine": "arm64", "system": "macOS 26",
        }
        path = "/tmp/control-preflight/invocation.json"
        parent_sha = "d" * 64
        derived = prepare.make_capture_launch_template(
            parent, path, parent_sha, self.schedule, "control")
        self.assertTrue(prepare.verify_capture_launch_template(
            parent, derived, path, parent_sha, self.schedule, "control"))
        self.assertEqual(derived["argv"], parent["argv"])
        self.assertEqual(derived["asset_sha256"], parent["asset_sha256"])
        self.assertEqual(derived["environment"]["PATH"], "/usr/bin")
        self.assertEqual(
            derived["environment"][prepare.CAPTURE_ENV_KEY],
            self.schedule["arms"]["control"]["environment_value"])
        for mutation in ("argv", "environment", "asset_sha256", "host"):
            changed = dict(derived)
            if mutation == "argv":
                changed["argv"] = list(derived["argv"]) + ["--unexpected"]
            elif mutation == "environment":
                changed["environment"] = dict(derived["environment"], NUMI_UNKNOWN_OVERRIDE="1")
            elif mutation == "asset_sha256":
                changed["asset_sha256"] = {"/tmp/native": "e" * 64}
            else:
                changed["host"] = "other host"
            with self.assertRaises(ValueError, msg="unapproved " + mutation + " change must fail closed"):
                prepare.verify_capture_launch_template(
                    parent, changed, path, parent_sha, self.schedule, "control")

    def test_parent_capture_selection_must_follow_pinned_017_cadence(self):
        parent = {"argv": ["/native", "--muscle-step-count", "10000",
                            "--muscle-step-seconds", "0.002"],
                  "environment": {prepare.CAPTURE_ENV_KEY: "0,31,63,95,127,159,191,223"}}
        self.assertEqual(prepare.validate_parent_capture_selection(parent),
                         "0,31,63,95,127,159,191,223")
        invalid = dict(parent, environment={prepare.CAPTURE_ENV_KEY: "0,32"})
        with self.assertRaisesRegex(ValueError, "not initial"):
            prepare.validate_parent_capture_selection(invalid)


if __name__ == "__main__":
    unittest.main()
