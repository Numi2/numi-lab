#!/usr/bin/env python3
import importlib.util
import struct
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "skin_rebind", Path(__file__).resolve().parents[1] / "tools/rebind_human_skin_reference.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(nb=4):
    nv, ni = 5, 6
    raw = bytearray(60 + nb*36 + nv*56 + ni*4 + nv*nb*4)
    struct.pack_into("<8s5I", raw, 0, b"NHSKIN1\0", 5, nb, nv, ni, 123)
    for i in range(nb):
        struct.pack_into("<I8f", raw, 60+i*36, 10+i, 0, 0, 0, 0, 0, 0, 1, 1)
    vo = 60+nb*36
    for i, xyz in enumerate([(0,0,0), (1,0,0), (0,1,0), (1,1,0), (10,10,0)]):
        struct.pack_into("<6f4I4f", raw, vo+i*56, *xyz, 0, 0, 1,
                         0, 0, 0, 0, 1, 0, 0, 0)
    io = vo+nv*56
    struct.pack_into("<6I", raw, io, 0, 1, 2, 1, 3, 2)
    for i in range(nv):
        struct.pack_into("<" + str(nb) + "f", raw, io+ni*4+i*nb*4,
                         *([1]+[0]*(nb-1)))
    return m.decode(bytes(raw))


class RebindTests(unittest.TestCase):
    def test_reference_preserved_and_unedited_bytes_exact(self):
        p = fixture()
        a = np.repeat(np.eye(3)[None], p["nb"], axis=0)
        b = np.array([[0,0,0], [.1,.2,.3], [0,0,0], [0,0,0]])
        w = p["weights"].copy()
        w[1] = [0,1,0,0]
        raw, report = m.rebind(p, w, a, b)
        q = m.decode(raw)
        np.testing.assert_allclose(m.positions(p, a, b), m.positions(q, a, b), atol=1e-7)
        self.assertEqual(report["changed_rows"], [1])
        self.assertEqual(raw[:p["vertoff"]], p["raw"][:p["vertoff"]])
        self.assertEqual(raw[p["indexoff"]:p["weightoff"]], p["raw"][p["indexoff"]:p["weightoff"]])
        for i in [0,2,3,4]:
            o = p["vertoff"]+i*56
            self.assertEqual(raw[o:o+56], p["raw"][o:o+56])
        np.testing.assert_array_equal(q["verts"][:,3:6], p["verts"][:,3:6])
        ids = np.frombuffer(raw, "<u4", 4, p["vertoff"]+56+24)
        self.assertEqual(int(ids[0]), 1)

    def test_patch_follows_topology_and_keeps_disconnected_vertex(self):
        p = fixture()
        w = m.local_rigid_weights(p, 11, [0], [1,.5,0])
        np.testing.assert_array_equal(w[0], [0,1,0,0])
        np.testing.assert_array_equal(w[1], [.5,.5,0,0])
        np.testing.assert_array_equal(w[2], [.5,.5,0,0])
        np.testing.assert_array_equal(w[3:], p["weights"][3:])

    def test_singular_inverse_rejected(self):
        p = fixture()
        a = np.repeat(np.eye(3)[None], 4, axis=0)
        a[1] = -a[0]
        w = p["weights"].copy()
        w[0] = [.5,.5,0,0]
        with self.assertRaisesRegex(m.RebindError, "conditioned"):
            m.rebind(p, w, a, np.zeros((4,3)))

    def test_negative_nonfinite_and_bad_partition_rejected(self):
        for row in [[-1,2,0,0], [np.nan,0,0,1], [.5,0,0,0]]:
            w = fixture()["weights"].copy()
            w[0] = row
            with self.assertRaises(m.RebindError):
                m.validate_weights(w)

    def test_identity_replacement_keeps_complete_file_exact(self):
        p = fixture()
        raw, report = m.rebind(p, p["weights"], np.repeat(np.eye(3)[None],4,0),
                               np.zeros((4,3)))
        self.assertEqual(raw, p["raw"])
        self.assertEqual(report["changed_rows"], [])

    def test_wrong_bindings_rejected(self):
        p = fixture()
        raw = bytearray(p["raw"])
        struct.pack_into("<I", raw, 60, 999)
        with self.assertRaisesRegex(m.RebindError, "identities"):
            m.replacement_weights(p, m.decode(bytes(raw)))

    def test_replacement_source_and_registration_must_match(self):
        p = fixture()
        for offset in [24, 28]:
            raw = bytearray(p["raw"])
            raw[offset] ^= 1
            with self.assertRaisesRegex(m.RebindError, "ancestry"):
                m.replacement_weights(p, m.decode(bytes(raw)))

    def test_bad_indices_and_truncation_rejected(self):
        p = fixture()
        raw = bytearray(p["raw"])
        struct.pack_into("<I", raw, p["indexoff"], p["nv"])
        for data in [bytes(raw), p["raw"][:-1]]:
            with self.assertRaises(m.RebindError):
                m.decode(data)

    def test_small_binding_count_top_four_padding(self):
        p = fixture(2)
        a = np.repeat(np.eye(3)[None],2,0)
        w = p["weights"].copy()
        w[0] = [.25,.75]
        raw, _ = m.rebind(p, w, a, np.zeros((2,3)))
        values = struct.unpack_from("<4I4f", raw, p["vertoff"]+24)
        self.assertEqual(values[:2], (1,0))
        np.testing.assert_allclose(values[4:], [.75,.25,0,0])

    def test_missing_and_duplicate_reference_poses_rejected(self):
        p = fixture()
        pose = {"source_poses": []}
        with self.assertRaises(m.RebindError):
            m.reference_transforms(p, pose, [0,0,0])
        row = {"core_body":10,"position":[0,0,0],"quaternion_xyzw":[0,0,0,1]}
        with self.assertRaisesRegex(m.RebindError, "duplicate"):
            m.reference_transforms(p, {"source_poses":[row,row]}, [0,0,0])

    def test_disjoint_repairs_compose_without_overwrite(self):
        p = fixture()
        a = np.repeat(np.eye(3)[None], 4, 0)
        b = np.array([[0,0,0], [.1,.2,.3], [0,0,0], [0,0,0]])
        repairs = []
        for row, weights in [(1, [0,1,0,0]), (3, [.5,0,.5,0])]:
            w = p["weights"].copy()
            w[row] = weights
            raw, _ = m.rebind(p, w, a, b)
            repairs.append(m.decode(raw))
        w, counts = m.merge_weight_repairs(p, repairs)
        self.assertEqual(counts, [1,1])
        np.testing.assert_array_equal(w[1], [0,1,0,0])
        np.testing.assert_array_equal(w[3], [.5,0,.5,0])
        raw, report = m.rebind(p, w, a, b)
        self.assertEqual(report["changed_rows"], [1,3])
        np.testing.assert_allclose(m.positions(m.decode(raw), a, b),
                                   m.positions(p, a, b), atol=1e-7)

    def test_conflicting_repairs_rejected_identical_overlap_allowed(self):
        p = fixture()
        a = np.repeat(np.eye(3)[None], 4, 0)
        repairs = []
        for weights in [[0,1,0,0], [0,0,1,0]]:
            w = p["weights"].copy()
            w[1] = weights
            raw, _ = m.rebind(p, w, a, np.zeros((4,3)))
            repairs.append(m.decode(raw))
        with self.assertRaisesRegex(m.RebindError, "vertex 1"):
            m.merge_weight_repairs(p, repairs)
        w, counts = m.merge_weight_repairs(p, [repairs[0], repairs[0]])
        np.testing.assert_array_equal(w, repairs[0]["weights"])
        self.assertEqual(counts, [1,1])
        np.testing.assert_array_equal(p["weights"], fixture()["weights"])

    def test_empty_composition_is_exact_identity(self):
        p = fixture()
        w, counts = m.merge_weight_repairs(p, [])
        np.testing.assert_array_equal(w, p["weights"])
        self.assertEqual(counts, [])

    def test_unknown_body_and_invalid_shell_rejected(self):
        p = fixture()
        for body, seeds, gates in [(999,[0],[1]), (10,[5],[1]), (10,[0],[1.1])]:
            with self.assertRaises(m.RebindError):
                m.local_rigid_weights(p, body, seeds, gates)


if __name__ == "__main__":
    unittest.main()
