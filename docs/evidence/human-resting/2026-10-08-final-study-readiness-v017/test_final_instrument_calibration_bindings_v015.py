#!/usr/bin/env python3
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_final_plan_v015", ROOT / "prepare_final_plan.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class FinalInstrumentCalibrationBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a = self.root / "instrument-a.json"
        self.b = self.root / "instrument-b.json"
        self.a.write_text("a\\n", encoding="utf-8")
        self.b.write_text("b\\n", encoding="utf-8")
        self.plan = {"instrument": {"artifacts": [str(self.a), str(self.b)]}}
        self.calibration = {"schema": "numi.science.calibration.v1", "bindings": {
            str(self.a): PREPARE.sha(self.a)
        }}

    def tearDown(self):
        self.temp.cleanup()

    def test_fills_late_added_artifact_and_returns_exact_final_set(self):
        result = PREPARE.complete_instrument_calibration_bindings(self.plan, self.calibration)
        self.assertEqual(set(result["bindings"]), set(self.plan["instrument"]["artifacts"]))
        self.assertEqual(result["bindings"][str(self.b)], PREPARE.sha(self.b))
        self.assertEqual(result["bindings"][str(self.a)], PREPARE.sha(self.a))

    def test_rejects_conflicting_existing_artifact_hash(self):
        self.calibration["bindings"][str(self.a)] = "0" * 64
        with self.assertRaisesRegex(ValueError, "conflicts with the final instrument artifact"):
            PREPARE.complete_instrument_calibration_bindings(self.plan, self.calibration)

    def test_rejects_binding_outside_final_instrument(self):
        self.calibration["bindings"][str(self.root / "stale.json")] = "1" * 64
        with self.assertRaisesRegex(ValueError, "outside the final instrument"):
            PREPARE.complete_instrument_calibration_bindings(self.plan, self.calibration)

    def test_rejects_duplicate_instrument_paths(self):
        self.plan["instrument"]["artifacts"].append(str(self.a))
        with self.assertRaisesRegex(ValueError, "duplicate paths"):
            PREPARE.complete_instrument_calibration_bindings(self.plan, self.calibration)

    def test_rejects_missing_late_added_artifact(self):
        self.plan["instrument"]["artifacts"].append(str(self.root / "missing.json"))
        with self.assertRaisesRegex(ValueError, "final instrument artifact must be a regular"):
            PREPARE.complete_instrument_calibration_bindings(self.plan, self.calibration)


if __name__ == "__main__":
    unittest.main()
