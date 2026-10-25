import csv
import hashlib
import importlib.util
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("load_transfer_075", HERE / "human_skin_fat_accepted_transfer.py")
adapter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(adapter)


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def payload_bytes(faces=(0, 1, 2)):
    nb, nv, ni = 1, 3, 3
    header = struct.pack("<8s5I", b"NHSKIN1\0", 5, nb, nv, ni, 0) + bytes(60 - 28)
    binding = struct.pack("<I8f", 0, *([0.0] * 8))
    vertices = b"".join(struct.pack("<14f", *([float(i), 0.0, 0.0] + [0.0] * 11)) for i in range(nv))
    indices = struct.pack("<3I", *faces)
    weights = struct.pack("<3f", 1.0, 1.0, 1.0)
    return header + binding + vertices + indices + weights


def make_fixture(root, *, transfer_edit=None, coarse_face_edit=None, payload_face=(0, 1, 2)):
    root = Path(root)
    transfer_fields = [
        "detail_row", "active", "detail_source_vertex_id", "detail_donor_body", "transfer_donor_body",
        "detail_rest_world_x_m", "detail_rest_world_y_m", "detail_rest_world_z_m",
        "coarse_vertex0", "coarse_vertex1", "coarse_vertex2", "weight0", "weight1", "weight2",
        "selected_face_row", "selected_face_stable_id",
    ]
    transfer_rows = [
        dict(detail_row=0, active=1, detail_source_vertex_id=0, detail_donor_body=4, transfer_donor_body=4,
             detail_rest_world_x_m=1.25, detail_rest_world_y_m=2.5, detail_rest_world_z_m=3.75,
             coarse_vertex0=0, coarse_vertex1=0, coarse_vertex2=1, weight0=0.25, weight1=0.5, weight2=0.25,
             selected_face_row=0, selected_face_stable_id=77),
        dict(detail_row=1, active=1, detail_source_vertex_id=1, detail_donor_body=4, transfer_donor_body=4,
             detail_rest_world_x_m=4.0, detail_rest_world_y_m=5.0, detail_rest_world_z_m=6.0,
             coarse_vertex0=0, coarse_vertex1=1, coarse_vertex2=2, weight0=0.2, weight1=0.3, weight2=0.5,
             selected_face_row=0, selected_face_stable_id=77),
        dict(detail_row=2, active=0, detail_source_vertex_id=2, detail_donor_body=9, transfer_donor_body=99,
             detail_rest_world_x_m=-1.0, detail_rest_world_y_m=-2.0, detail_rest_world_z_m=-3.0,
             coarse_vertex0=999, coarse_vertex1=-1, coarse_vertex2=999, weight0="nan", weight1=-2.0, weight2=5.0,
             selected_face_row=999, selected_face_stable_id=-1),
    ]
    if transfer_edit:
        transfer_edit(transfer_rows)
    write_csv(root / "transfer.csv", transfer_fields, transfer_rows)
    coarse_fields = ["coarse_row", "world_x_m", "world_y_m", "world_z_m"]
    coarse_rows = [
        dict(coarse_row=0, world_x_m=0.0, world_y_m=0.0, world_z_m=0.0),
        dict(coarse_row=1, world_x_m=1.0, world_y_m=0.0, world_z_m=0.0),
        dict(coarse_row=2, world_x_m=0.0, world_y_m=1.0, world_z_m=0.0),
        dict(coarse_row=3, world_x_m=1.0, world_y_m=1.0, world_z_m=0.0),
    ]
    write_csv(root / "coarse.csv", coarse_fields, coarse_rows)
    face_fields = ["coarse_face_row", "stable_source_face_id", "vertex0", "vertex1", "vertex2"]
    coarse_face = dict(coarse_face_row=0, stable_source_face_id=77, vertex0=0, vertex1=1, vertex2=2)
    if coarse_face_edit:
        coarse_face_edit(coarse_face)
    write_csv(root / "coarse-faces.csv", face_fields, [coarse_face])
    (root / "detail.nhskin").write_bytes(payload_bytes(payload_face))
    paths = {"transfer_rows": root / "transfer.csv", "coarse_reference": root / "coarse.csv",
             "coarse_faces": root / "coarse-faces.csv", "detail_payload": root / "detail.nhskin"}
    pins = {key: adapter.sha256_file(value) for key, value in paths.items()}
    return paths, pins


class Transfer075Tests(unittest.TestCase):
    def fixture(self, **kwargs):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return make_fixture(temp.name, **kwargs)

    def load(self, paths, pins):
        return adapter.load_transfer075(paths["transfer_rows"], paths["coarse_reference"],
            paths["coarse_faces"], paths["detail_payload"], expected_sha256=pins)

    def test_active_inactive_duplicate_columns_and_detail_faces(self):
        paths, pins = self.fixture()
        W, detail, coarse, faces = self.load(paths, pins)
        self.assertEqual(W.shape, (3, 4))
        self.assertEqual(W.dtype, np.float32)
        self.assertEqual(W.getrow(0).indices.tolist(), [0, 1])
        np.testing.assert_allclose(W.getrow(0).data, np.array([0.75, 0.25], dtype=np.float32), rtol=0, atol=0)
        np.testing.assert_allclose(W.getrow(1).toarray(), [[0.2, 0.3, 0.5, 0.0]], rtol=0, atol=1e-7)
        self.assertEqual(W.getrow(2).nnz, 0)
        self.assertEqual(detail.dtype, np.float32)
        np.testing.assert_array_equal(detail[0], np.array([1.25, 2.5, 3.75], dtype=np.float32))
        self.assertEqual(coarse.shape, (4, 3))
        np.testing.assert_array_equal(faces, np.array([[0, 1, 2]], dtype=np.uint32))

    def test_hash_pins_fail_closed(self):
        paths, pins = self.fixture()
        pins = dict(pins)
        pins["detail_payload"] = "0" * 64
        with self.assertRaisesRegex(adapter.Transfer075Error, "SHA-256 mismatch"):
            self.load(paths, pins)

    def test_invalid_selected_face_row_rejected(self):
        def edit(rows): rows[0]["selected_face_row"] = 1
        paths, pins = self.fixture(transfer_edit=edit)
        with self.assertRaisesRegex(adapter.Transfer075Error, "selected coarse face row"):
            self.load(paths, pins)

    def test_stable_face_identity_mismatch_rejected(self):
        def edit(rows): rows[0]["selected_face_stable_id"] = 78
        paths, pins = self.fixture(transfer_edit=edit)
        with self.assertRaisesRegex(adapter.Transfer075Error, "stable face identity"):
            self.load(paths, pins)

    def test_support_outside_selected_face_rejected(self):
        def edit(rows): rows[0]["coarse_vertex0"] = 3
        paths, pins = self.fixture(transfer_edit=edit)
        with self.assertRaisesRegex(adapter.Transfer075Error, "support is not contained"):
            self.load(paths, pins)

    def test_negative_nonfinite_and_nonpartition_weights_rejected(self):
        for key, value, pattern in (("weight0", -0.01, "nonnegative"),
                                    ("weight0", "nan", "nonfinite"),
                                    ("weight0", 0.1, "do not sum")):
            with self.subTest(key=key, value=value):
                def edit(rows, k=key, v=value):
                    rows[0][k] = v
                paths, pins = self.fixture(transfer_edit=edit)
                with self.assertRaisesRegex(adapter.Transfer075Error, pattern):
                    self.load(paths, pins)

    def test_coarse_face_out_of_range_rejected(self):
        def edit(face): face["vertex2"] = 4
        paths, pins = self.fixture(coarse_face_edit=edit)
        with self.assertRaisesRegex(adapter.Transfer075Error, "coarse face table"):
            self.load(paths, pins)

    def test_detailed_payload_face_out_of_range_rejected(self):
        paths, pins = self.fixture(payload_face=(0, 1, 3))
        with self.assertRaisesRegex(adapter.Transfer075Error, "detail NHSKIN face index"):
            self.load(paths, pins)

    def test_repeated_detail_face_vertex_rejected(self):
        paths, pins = self.fixture(payload_face=(0, 1, 1))
        with self.assertRaisesRegex(adapter.Transfer075Error, "repeated face vertex"):
            self.load(paths, pins)

    def test_active_donor_mismatch_rejected(self):
        def edit(rows): rows[0]["transfer_donor_body"] = 5
        paths, pins = self.fixture(transfer_edit=edit)
        with self.assertRaisesRegex(adapter.Transfer075Error, "donor differs"):
            self.load(paths, pins)


if __name__ == "__main__":
    unittest.main()
