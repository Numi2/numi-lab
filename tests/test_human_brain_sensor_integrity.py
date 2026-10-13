import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import extract_human_brain_committed_transitions as transitions
import human_brain_training_cohort as cohort
import human_brain_recovery_audit as recovery

EVIDENCE = ROOT / "docs/media/human-brain-launcher-recovery-20261013"


def rows(name):
    return [line for line in (EVIDENCE / name).read_text().splitlines()
            if line.startswith("human_brain_sensor_audit=accepted")]


class SensorIntegrityTests(unittest.TestCase):
    def test_native_fixed_rows_are_admitted_and_original_stride_rows_rejected(self):
        fixed = rows("standing-touch-coverage-excerpts.log")
        original = rows("standing-accepted-audit-smoke.log")
        self.assertEqual(len(fixed), 4)
        self.assertEqual(len(original), 4)
        for step, line in enumerate(fixed, 1):
            frame = transitions.sensor(line, step)
            self.assertEqual(len(frame["touchNormalForceN"]), 10)
        for step, line in enumerate(original, 1):
            with self.assertRaisesRegex(ValueError, "required native sensory"):
                transitions.sensor(line, step)

    def test_corrupted_native_head_force_and_clock_reject(self):
        line = rows("standing-touch-coverage-excerpts.log")[0]
        corruptions = [
            line.replace("head_validity=983040", "head_validity=0"),
            re.sub(r"head_quaternion_xyzw=\[[^]]+\]", "head_quaternion_xyzw=[0,0,0,0]", line),
            line.replace("receptor_timestamp_us=1000", "receptor_timestamp_us=2000"),
            line.replace("touch_normal_force_n=[0,0,", "touch_normal_force_n=[-1,0,"),
            line.replace("touch_normal_force_n=[0,0,", "touch_normal_force_n=[nan,0,"),
        ]
        self.assertTrue(all(changed != line for changed in corruptions))
        for changed in corruptions:
            with self.assertRaises(ValueError):
                transitions.sensor(changed, 1)

    def test_cohort_reports_coverage_separately_from_physical_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.log"
            for name, valid in [("standing-touch-coverage-excerpts.log", True),
                                ("standing-accepted-audit-smoke.log", False)]:
                path.write_text("\n".join(rows(name)) + "\n")
                result = cohort.receipt(path, 4, 0, True)
                self.assertIs(result["sensorCoverageValidated"], valid)
                self.assertFalse(result["accepted"])  # Sensory rows alone are no physical episode.
                self.assertEqual(result["invalidSensorSteps"], [] if valid else [1, 2, 3, 4])

    def test_recovery_reader_rejects_incomplete_native_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            path.joinpath("launch.log").write_text(rows("standing-accepted-audit-smoke.log")[0] + "\n")
            # Source-manifest admission has separate tests and precedes this reader.
            with patch.object(recovery, "manifest", return_value=([], {})):
                with self.assertRaisesRegex(ValueError, "required native sensory"):
                    recovery.read_run(path, 4)


if __name__ == "__main__":
    unittest.main()
