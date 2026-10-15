"""GPU checks for existing-kernel skin replay; set BIN and METALLIB explicitly.

These synthetic ABI fixtures test replay plumbing and rejection boundaries.
They do not qualify anatomical geometry, mechanics or physiology.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

BIN = os.environ.get("NUMI_RESTING_SKIN_REPLAY_BIN")
LIB = os.environ.get("NUMI_RESTING_SKIN_REPLAY_METALLIB")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(BIN and LIB, "explicit native replay binary and existing Metal library required")
class ExistingSkinKernelReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.library = Path(LIB)

    def tearDown(self):
        self.temp.cleanup()

    def pinned(self, path):
        return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))

    def fixture(self, *, tiled=False, breathing=False):
        stride = 32 if tiled else 1
        influence = bytearray((5 * stride + 1) * 48)
        weights = [.125, .125, .125, .125, .25, .25]
        for j, w in enumerate(weights):
            struct.pack_into("<8f4I", influence, j * stride * 48,
                             0, 0, 0, w, 0, 0, 1, 0, int(j >= 4), 0, 0, 0)
        ip = self.root / "influences.bin"
        ip.write_bytes(influence)
        mp = self.root / "map.bin"
        mp.write_bytes(struct.pack("<4I8f", 13, 6, 3, 0, 1, 0, 0, 0, 0, 0, 0, 1))
        anatomy = bytearray(1008)
        struct.pack_into("<4I", anatomy, 0, 0, int(breathing), 0, 0)
        struct.pack_into("<4f", anatomy, 16, 0, 0, 0, .0025)
        struct.pack_into("<4f", anatomy, 32, 0, 0, 1, 1)
        struct.pack_into("<4f", anatomy, 96, 0, 0, .1, 0)
        ap = self.root / "anatomy.bin"
        ap.write_bytes(anatomy)
        source = dict(synthetic_fixture=True, skin_source_mapping=dict(
            source_vertex_count=1, vertex_map_record_stride_bytes=48,
            influence_record_stride_bytes=48, influence_first_index=13,
            influence_layout="tile32" if tiled else "vertex-major",
            vertex_map=self.pinned(mp), influences=self.pinned(ip),
            anatomy_parameters=self.pinned(ap), kernel_mode="native-library",
            kernel=self.pinned(self.library)))
        pose = dict(synthetic_fixture=True, accepted_body_poses=[
            dict(body_index=0, position_m=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1]),
            dict(body_index=1, position_m=[2, 0, 0], quaternion_xyzw=[0, 0, 0, 1])],
            accepted_respiratory_motion=dict(
                diaphragm_swept_volume_m3=.0001 if breathing else 0,
                rib_swept_volume_m3=.0002 if breathing else 0,
                diaphragm_volume_velocity_m3_s=0, rib_volume_velocity_m3_s=0))
        return source, pose

    def invoke(self, source, pose, *, source_hash=None):
        sp, pp = self.root / "source.json", self.root / "pose.json"
        sp.write_text(json.dumps(source))
        pp.write_text(json.dumps(pose))
        out = self.root / "output"
        run = subprocess.run([BIN, str(sp), source_hash or digest(sp),
                              str(pp), digest(pp), str(out)],
                             text=True, capture_output=True, timeout=90)
        return run, out

    def test_more_than_four_positive_weights_and_nonzero_buffer_base(self):
        source, pose = self.fixture()
        run, out = self.invoke(source, pose)
        self.assertEqual(run.returncode, 0, run.stderr)
        position = struct.unpack_from("<4f", (out / "skin-vertices.bin").read_bytes())
        self.assertEqual(position, (1, 0, 0, 1))
        report = json.loads((out / "receipt.json").read_text())
        self.assertFalse(report["physical_stepping"])
        self.assertFalse(report["anatomical_admission"])

    def test_tile32_uses_every_positive_source_weight(self):
        source, pose = self.fixture(tiled=True)
        run, out = self.invoke(source, pose)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(struct.unpack_from("<4f", (out / "skin-vertices.bin").read_bytes()),
                         (1, 0, 0, 1))

    def test_existing_respiratory_field_responds_to_both_swept_volumes(self):
        source, pose = self.fixture(breathing=True)
        run, out = self.invoke(source, pose)
        self.assertEqual(run.returncode, 0, run.stderr)
        x, y, z, w = struct.unpack_from("<4f", (out / "skin-vertices.bin").read_bytes())
        self.assertAlmostEqual(x, math.sqrt(.0028 / .0026), delta=3e-7)
        self.assertEqual(y, 0)
        self.assertAlmostEqual(z, -.001, delta=1e-9)
        self.assertEqual(w, 1)

    def test_changed_source_receipt_pin_rejects_before_output(self):
        source, pose = self.fixture()
        run, out = self.invoke(source, pose, source_hash="0" * 64)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("receipt hash mismatch", run.stderr)
        self.assertFalse(out.exists())

    def test_out_of_range_influence_rejects_before_output(self):
        source, pose = self.fixture()
        path = Path(source["skin_source_mapping"]["vertex_map"]["path"])
        b = bytearray(path.read_bytes())
        struct.pack_into("<I", b, 0, 1000)
        path.write_bytes(b)
        source["skin_source_mapping"]["vertex_map"] = self.pinned(path)
        run, out = self.invoke(source, pose)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("influence index out of range", run.stderr)
        self.assertFalse(out.exists())

    def test_changed_influence_bytes_reject_before_output(self):
        source, pose = self.fixture()
        path = Path(source["skin_source_mapping"]["influences"]["path"])
        b = bytearray(path.read_bytes())
        struct.pack_into("<f", b, 12, .5)
        path.write_bytes(b)
        run, out = self.invoke(source, pose)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("differs from receipt pin", run.stderr)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
