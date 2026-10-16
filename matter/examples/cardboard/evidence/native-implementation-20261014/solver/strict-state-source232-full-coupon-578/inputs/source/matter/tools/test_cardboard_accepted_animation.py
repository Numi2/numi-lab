import csv
import json
from pathlib import Path
import tempfile
import unittest

from cardboard_accepted_animation import coupon_frames, read_obj


class AcceptedGeometry(unittest.TestCase):
    def fixture(self, root, accepted=1, status=0, complete=True):
        solver = dict(steps=1, hold_steps=0, unload_steps=0, relax_steps=0,
                      release_steps=0, relative_residual_tolerance=1e-6,
                      volume_tolerance=1e-4, pressure_tolerance=1e-4)
        (root / "manifest.json").write_text(json.dumps(dict(solver=solver)))
        row = dict(environment=0, step=0, time_s=1e-3, phase="bend", arm="bent",
                   status_code=status, step_accepted=accepted,
                   certificate_residual=1e-7, certificate_volume=0,
                   certificate_pressure=0, certificate_raw_accepted_flag=1)
        with (root / "observations.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader(); writer.writerow(row)
        if complete:
            (root / "result.json").write_text(json.dumps(dict(status="completed")))
        geometry = root / "accepted_step_000001_bent.obj"
        geometry.write_text("# native state\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        return geometry

    def test_complete_certificate_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            frames, evidence = coupon_frames(root, 0, False)
            self.assertEqual(len(frames), 1)
            self.assertTrue(evidence["complete"])

    def test_accepted_filename_does_not_override_rejected_certificate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root, accepted=0, status=10)
            with self.assertRaisesRegex(ValueError, "incomplete or rejected"):
                coupon_frames(root, 0, False)
            with self.assertRaisesRegex(ValueError, "no certificate-accepted"):
                coupon_frames(root, 0, True)

    def test_missing_final_result_requires_diagnostic_label(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root, complete=False)
            with self.assertRaisesRegex(ValueError, "incomplete or rejected"):
                coupon_frames(root, 0, False)
            _, evidence = coupon_frames(root, 0, True)
            self.assertFalse(evidence["complete"])

    def test_invalid_topology_and_no_silent_decimation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = self.fixture(root)
            self.assertEqual(len(read_obj(path, 1)[1]), 1)
            with self.assertRaisesRegex(ValueError, "budget exceeded"):
                read_obj(path, 0)
            path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n")
            with self.assertRaisesRegex(ValueError, "invalid OBJ"):
                read_obj(path, 1)


if __name__ == "__main__":
    unittest.main()
