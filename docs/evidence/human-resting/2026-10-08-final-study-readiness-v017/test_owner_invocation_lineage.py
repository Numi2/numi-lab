#!/usr/bin/env python3
import csv
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("readiness_analysis", str(ROOT / "analyze_final_pair.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class OwnerInvocationLineageTests(unittest.TestCase):
    def test_wrapper_schema_matches_observation_emitted_by_frozen_owner(self):
        owner = analysis.prepare.owner
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "owner-observation-schema-fixture.csv"
            fields = list(owner.TRACE_COLUMNS)
            with trace.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index in range(1241):
                    row = {key: 0.0 for key in fields}
                    row["time_s"] = index * 0.25
                    writer.writerow(row)
            args = Namespace(steps=155000, dt=0.002, window_s=30.0,
                             start_s=60.0, end_s=100.0, unit_id="schema-fixture",
                             arm="control", scale=0.5)
            native = {"accepted_steps": 155000, "simulated_s": 310.0,
                      "device": "Apple M4 Pro", "world_fingerprint": 1,
                      "vascular_dense45": True, "brain_control": False,
                      "real_time_factor": 0.1}
            observation = owner.observation(args, trace, native, "fixture log")
        self.assertEqual(observation["schema"], "numi.human-resting.intervention-observation.v1")
        self.assertEqual(observation["schema"], analysis.OWNER_OBSERVATION_SCHEMA)

    def test_frozen_human_runtime_helper_verifies_a_retained_native_scene_log(self):
        self.assertEqual(analysis.prepare.sha(analysis.prepare.HUMAN_RESTING_RUN),
                         analysis.prepare.HUMAN_RESTING_RUN_SHA)
        metadata = json.loads((analysis.prepare.E / "native-source-state-cycle-914/run-metadata.json").read_text())
        invocation = json.loads((analysis.prepare.E / "native-source-state-cycle-914/invocation.json").read_text())
        proof = analysis.verify_registered_scene_loaded_runtime(
            analysis.prepare.TERMINAL_931_NATIVE_LOG, metadata, invocation)
        self.assertTrue(proof["verified"])
        self.assertEqual(proof["expected_path"], str(analysis.prepare.LIBMETALROBO.resolve()))
        self.assertEqual(proof["expected_sha256"], analysis.prepare.LIBMETALROBO_SHA)
        self.assertEqual(len(proof["observed_images"]), 1)

    def test_frozen_owner_serializes_derived_template_hash_in_both_outputs(self):
        owner_source = analysis.prepare.OWNER.read_text(encoding="utf-8")
        self.assertIn('"reference_invocation_sha256": sha256_file(invocation_path)', owner_source)
        self.assertIn('reference_invocation_sha256=sha256_file(invocation_path)', owner_source)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent_path = root / "parent-invocation.json"
            template_path = root / "derived-template.json"
            schedule = analysis.prepare.accepted_geometry_capture_schedule()
            parent = {"argv": ["/native", "--muscle-step-count", "10000",
                               "--muscle-step-seconds", "0.002"],
                      "environment": {"PATH": "/usr/bin",
                                      analysis.prepare.CAPTURE_ENV_KEY:
                                          "0,31,63,95,127,159,191,223"},
                      "asset_sha256": {"/native": "a" * 64}}
            parent_path.write_text(json.dumps(parent), encoding="utf-8")
            parent_sha = analysis.audit.sha256(parent_path)
            template = analysis.prepare.make_capture_launch_template(
                parent, parent_path, parent_sha, schedule, "control")
            template_path.write_text(json.dumps(template), encoding="utf-8")
            template_sha = analysis.audit.sha256(template_path)
            self.assertNotEqual(template_sha, parent_sha)

            # These fields follow the frozen owner writer: invocation_path is the
            # --invocation argument, which is the registered derived template.
            scene_invocation = {"reference_invocation_sha256": template_sha,
                                "asset_sha256": parent["asset_sha256"]}
            observation = {"reference_invocation_sha256": template_sha}
            self.assertEqual(analysis.verify_owner_scene_invocation_binding(
                scene_invocation, observation, template_path), template_sha)
            scene_invocation["reference_invocation_sha256"] = parent_sha
            with self.assertRaisesRegex(ValueError, "derived launch-template"):
                analysis.verify_owner_scene_invocation_binding(
                    scene_invocation, observation, template_path)
            scene_invocation["reference_invocation_sha256"] = template_sha
            observation["reference_invocation_sha256"] = parent_sha
            with self.assertRaisesRegex(ValueError, "owner observation"):
                analysis.verify_owner_scene_invocation_binding(
                    scene_invocation, observation, template_path)


if __name__ == "__main__":
    unittest.main()
