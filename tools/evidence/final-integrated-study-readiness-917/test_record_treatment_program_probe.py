#!/usr/bin/env python3
"""CPU-only construction and failure-gate tests for the native identity recorder."""
import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "record_treatment_program_probe", str(ROOT / "record_treatment_program_probe.py"))
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)
readiness = recorder.load_readiness()


def control_invocation(intervention=False):
    argv = [
        str(readiness.NATIVE_BINARY),
        "--resting-scene", "/evidence/scene.json",
        "unused-position-3",
        "/evidence/control",
        "--resting-movie", "/evidence/control/native-viewer.mov",
        "--resting-anatomy-receipt", "/evidence/resting-anatomy-receipt.json",
        "--torso-anatomy-payload", "/evidence/NHANATOMY",
        "--muscle-step-count", "10000",
        "--muscle-step-seconds", "0.002",
    ]
    if intervention:
        argv.extend(["--resting-drive-intervention", "0", "20", "0.5"])
    return {
        "argv": argv,
        "environment": {
            "DYLD_LIBRARY_PATH": "/build/lib:/build/matter",
            "DYLD_PRINT_LIBRARIES": "1",
            "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT": "/evidence/control/failure.json",
        },
    }


class IdentityProbeRecorderTests(unittest.TestCase):
    def test_treatment_is_appended_beyond_twenty_second_probe(self):
        scene = Path("/evidence/final-control").resolve()
        output = scene / recorder.PROGRAM_PROBE_NAME
        argv, env = recorder.make_probe(control_invocation(), scene, output, readiness)
        index = argv.index("--resting-drive-intervention")
        self.assertEqual([float(x) for x in argv[index + 1:index + 4]], [60.0, 100.0, 0.5])
        self.assertEqual(index + 4, len(argv))
        self.assertEqual(int(argv[argv.index("--muscle-step-count") + 1]), 10000)
        self.assertEqual(float(argv[argv.index("--muscle-step-seconds") + 1]), 0.002)
        self.assertGreater(100.0, 10000 * 0.002)
        self.assertEqual(Path(argv[4]), output)
        self.assertEqual(Path(argv[argv.index("--resting-movie") + 1]),
                         output / "native-viewer.mov")
        self.assertEqual(env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"],
                         str(output / "common-field-failure.json"))

    def test_rejects_control_with_preexisting_intervention(self):
        scene = Path("/evidence/final-control").resolve()
        with self.assertRaisesRegex(ValueError, "unexpectedly includes"):
            recorder.make_probe(control_invocation(intervention=True), scene,
                                scene / recorder.PROGRAM_PROBE_NAME, readiness)

    def test_nonzero_native_exit_is_preserved(self):
        self.assertEqual(recorder.recorder_exit_status(17, None, [], [], True), 17)

    def test_zero_native_exit_does_not_hide_failed_validation(self):
        self.assertEqual(recorder.recorder_exit_status(0, "missing terminal", [], [], True), 1)
        self.assertEqual(recorder.recorder_exit_status(0, None, ["/src/a"], [], True), 1)
        self.assertEqual(recorder.recorder_exit_status(0, None, [], ["/assets/a"], True), 1)
        self.assertEqual(recorder.recorder_exit_status(0, None, [], [], False), 1)

    def test_matching_probe_has_zero_recorder_status(self):
        self.assertEqual(recorder.recorder_exit_status(0, None, [], [], True), 0)

    def test_native_environment_drops_unrecorded_prefixes_and_preserves_path(self):
        tracked={"DYLD_LIBRARY_PATH":"/pinned/lib:/pinned/matter",
                 "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT":"0"}
        inherited={"PATH":"/usr/bin:/bin","HOME":"/Users/n",
                   "NUMI_UNDECLARED_EXPERIMENT":"1","DYLD_INSERT_LIBRARIES":"/tmp/x"}
        env=recorder.make_native_environment(inherited,tracked)
        self.assertEqual(env["PATH"],inherited["PATH"])
        self.assertEqual(env["HOME"],inherited["HOME"])
        self.assertNotIn("NUMI_UNDECLARED_EXPERIMENT",env)
        self.assertNotIn("DYLD_INSERT_LIBRARIES",env)
        self.assertEqual(recorder.resting_run.invocation_environment(env),tracked)

    def test_pinned_parser_accepts_future_interval_without_run_duration_bound(self):
        evidence = recorder.verify_future_interval_parser(readiness)
        self.assertTrue(evidence["no_run_duration_bound_in_parser"])
        self.assertEqual(evidence["source_predicate"],
                         "start >= 0; end > start; 0 <= scale <= 2")
        self.assertEqual(evidence["sha256"], readiness.sha(readiness.PROBE_SOURCE))

    def test_missing_control_fails_before_creating_probe_output(self):
        scene = Path("/tmp/numi-resting-identity-recorder-missing-control")
        output = scene / recorder.PROGRAM_PROBE_NAME
        with self.assertRaisesRegex(ValueError, "control preflight directory is missing"):
            recorder.main(["--scene-preflight-dir", str(scene),
                           "--anatomy-receipt", "/tmp/missing-anatomy-receipt.json"])
        self.assertFalse(output.exists())

    def test_readiness_prints_recorder_command_not_raw_native_command(self):
        import contextlib
        import io
        scene = Path("/tmp/numi-final-control-command")
        control = control_invocation()
        predicted = {
            "treatment_program_fingerprint_predicted": 123456,
            "treatment_body_source_fingerprint_predicted": 654321,
        }
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            readiness.print_probe_command(control, scene, predicted)
        output = stream.getvalue()
        self.assertIn(str(recorder.ROOT / "record_treatment_program_probe.py"), output)
        self.assertIn("--scene-preflight-dir", output)
        self.assertIn("--anatomy-receipt /evidence/resting-anatomy-receipt.json", output)
        self.assertIn("--execute", output)
        self.assertNotIn("Recorded native payload command", output)


if __name__ == "__main__":
    unittest.main()
