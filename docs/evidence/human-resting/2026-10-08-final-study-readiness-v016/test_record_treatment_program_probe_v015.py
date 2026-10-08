#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "record_treatment_program_probe_v015", ROOT / "record_treatment_program_probe.py")
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)
readiness = recorder.load_readiness()


def control_invocation(intervention=False, respiration="/selected/final-respiration.json"):
    argv = [
        str(readiness.SCENE_NATIVE_BINARY), "--resting-scene", "/assets/circulation.json", respiration,
        "unused-position-3", "/evidence/control",
        "--resting-movie", "/evidence/control/native-viewer.mov",
        "--resting-anatomy-receipt", "/evidence/composed-receipt.json",
        "--torso-anatomy-payload", "/evidence/final.nhanatomy",
        "--muscle-step-count", "10000", "--muscle-step-seconds", "0.002",
    ]
    if intervention:
        argv.extend(["--resting-drive-intervention", "0", "20", "0.5"])
    return {"argv": argv, "environment": {
        "DYLD_LIBRARY_PATH": "/build014/lib:/build018/matter",
        "DYLD_PRINT_LIBRARIES": "1",
        "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT": "/evidence/control/failure.json",
    }}


class V015IdentityProbeRecorderTests(unittest.TestCase):
    def test_probe_child_is_versioned_and_not_reused_from_v010(self):
        self.assertEqual(recorder.PROGRAM_PROBE_NAME, "treatment-program-probe-v015")
        self.assertEqual(readiness.PROGRAM_PROBE_NAME, recorder.PROGRAM_PROBE_NAME)
        self.assertNotEqual(recorder.PROGRAM_PROBE_NAME, "treatment-program-probe-v010")

    def test_future_intervention_changes_only_program_and_output_paths(self):
        scene = Path("/evidence/final-control").resolve()
        output = scene / recorder.PROGRAM_PROBE_NAME
        control = control_invocation(respiration="/caller/final-respiration.json")
        argv, env = recorder.make_probe(control, scene, output, readiness)
        self.assertEqual(argv[0], str(readiness.SCENE_NATIVE_BINARY))
        self.assertNotEqual(argv[0], str(readiness.NATIVE_BINARY))
        self.assertEqual(argv[argv.index("--resting-scene") + 2],
                         "/caller/final-respiration.json")
        index = argv.index("--resting-drive-intervention")
        self.assertEqual([float(x) for x in argv[index + 1:index + 4]], [60.0, 100.0, 0.5])
        self.assertEqual(index + 4, len(argv))
        self.assertEqual(Path(argv[4]), output)
        self.assertEqual(Path(argv[argv.index("--resting-movie") + 1]),
                         output / "native-viewer.mov")
        self.assertEqual(env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"],
                         str(output / "common-field-failure.json"))

    def test_probe_refuses_already_treated_control(self):
        scene = Path("/evidence/final-control").resolve()
        with self.assertRaisesRegex(ValueError, "unexpectedly includes"):
            recorder.make_probe(control_invocation(intervention=True), scene,
                                scene / recorder.PROGRAM_PROBE_NAME, readiness)

    def test_decimal_uint64_record_normalizes_without_precision_loss(self):
        self.assertEqual(recorder.normalize_uint64_identity(
            "13712686933603975462", "fingerprint"), 13712686933603975462)
        for value in (True, 1.0, -1, "01", "+1", "18446744073709551616"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                recorder.normalize_uint64_identity(value, "fingerprint")


if __name__ == "__main__":
    unittest.main()
