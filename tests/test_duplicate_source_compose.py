import importlib.util
from pathlib import Path
import struct
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("duplicate_compose", ROOT / "tools" / "rebind_human_skin_reference.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    nb, nv, nf = 4, 5, 2
    raw = bytearray(60 + nb * 36 + nv * 56 + nf * 12 + nv * nb * 4)
    struct.pack_into("<8s5I", raw, 0, b"NHSKIN1\0", 5, nb, nv, nf * 3, 123)
    for i in range(nb):
        struct.pack_into("<I8f", raw, 60 + i * 36, 10 + i, 0, 0, 0, 0, 0, 0, 1, 1)
    vo = 60 + nb * 36
    xyzs = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (10, 10, 0)]
    for i, xyz in enumerate(xyzs):
        struct.pack_into("<6f4I4f", raw, vo + i * 56, *xyz, 0, 0, 1,
                         0, 1, 2, 3, 1, 0, 0, 0)
    io = vo + nv * 56
    struct.pack_into("<6I", raw, io, 0, 1, 2, 1, 3, 2)
    for i in range(nv):
        struct.pack_into("<4f", raw, io + nf * 12 + i * nb * 4, 1, 0, 0, 0)
    return m.decode(bytes(raw))


def duplicate_patch(base, parent, source):
    n, nf = base["nv"], len(base["inds"])
    outrow = n
    verts = np.vstack((base["verts"], base["verts"][source].copy()))
    verts[outrow, 0] += 0.002
    weights = np.vstack((base["weights"], base["weights"][source]))
    faces = base["inds"].astype(np.int64).copy()
    corner = int(np.flatnonzero(faces[parent] == source)[0])
    faces[parent, corner] = outrow
    header = bytearray(base["raw"][:60])
    struct.pack_into("<2I", header, 16, n + 1, nf * 3)
    raw = (bytes(header) + base["binds"].tobytes() + verts.astype("<f4").tobytes() +
           faces.astype("<u4").tobytes() + weights.astype("<f4").tobytes())
    other = m.decode(raw)
    ancestry_offsets = np.arange(n + 2, dtype=np.int64)
    source_ids = np.r_[np.arange(n, dtype=np.int64), source]
    source_weights = np.ones(n + 1, dtype=np.float64)
    return dict(payload=other,
                face_parent_source_rows=np.arange(nf, dtype=np.int64),
                protected_source_face_ids=np.asarray([i for i in range(nf) if i != parent], dtype=np.int64),
                replaced_source_face_ids=np.asarray([parent], dtype=np.int64),
                duplicated_output_vertex_ids=np.asarray([outrow], dtype=np.int64),
                duplicated_source_vertex_ids=np.asarray([source], dtype=np.int64),
                ancestry_offsets=ancestry_offsets, source_vertex_ids=source_ids,
                source_weights=source_weights)


class DuplicateSourceComposition(unittest.TestCase):
    def test_disjoint_duplicate_regions_compose_and_keep_originals(self):
        base = fixture()
        left, right = duplicate_patch(base, 0, 0), duplicate_patch(base, 1, 3)
        raw, report = m.merge_source_topology_repairs(base, duplicate_repairs=[left, right])
        out = m.decode(raw)
        self.assertEqual(out["nv"], 7)
        self.assertEqual(len(out["inds"]), 2)
        self.assertEqual(report["face_parent_source_rows"], [0, 1])
        self.assertEqual(report["added_duplicate_vertices"], 2)
        self.assertFalse(report["physical_admission"])
        self.assertFalse(report["geometric_overlap_certified"])
        np.testing.assert_array_equal(out["verts"][:5, 6:], base["verts"][:, 6:])
        np.testing.assert_array_equal(out["weights"][:5], base["weights"])
        np.testing.assert_array_equal(out["inds"], [[5, 1, 2], [1, 6, 2]])

    def test_duplicate_source_can_repeat_but_parent_rows_must_be_disjoint(self):
        base = fixture()
        a, b = duplicate_patch(base, 0, 1), duplicate_patch(base, 1, 1)
        raw, report = m.merge_source_topology_repairs(base, duplicate_repairs=[a, b])
        self.assertEqual(m.decode(raw)["nv"], 7)
        self.assertEqual(report["added_duplicate_vertices"], 2)
        with self.assertRaisesRegex(m.RebindError, "overlapping regional"):
            m.merge_source_topology_repairs(base, duplicate_repairs=[a, duplicate_patch(base, 0, 0)])

    def test_bad_source_id_csr_or_binding_fails_closed(self):
        base = fixture()
        repair = duplicate_patch(base, 0, 0)
        bad = dict(repair)
        bad["source_weights"] = np.r_[np.ones(base["nv"]), 0.5]
        with self.assertRaisesRegex(m.RebindError, "direct unit-weight"):
            m.merge_source_topology_repairs(base, duplicate_repairs=[bad])
        bad = dict(repair)
        bad["duplicated_source_vertex_ids"] = np.asarray([3], dtype=np.int64)
        with self.assertRaisesRegex(m.RebindError, "direct unit-weight"):
            m.merge_source_topology_repairs(base, duplicate_repairs=[bad])

    def test_parent_map_and_compact_source_binding_are_verified(self):
        base = fixture()
        repair = duplicate_patch(base, 0, 0)
        bad = dict(repair)
        bad["face_parent_source_rows"] = np.asarray([1, 0], dtype=np.int64)
        with self.assertRaisesRegex(m.RebindError, "parent identity"):
            m.merge_source_topology_repairs(base, duplicate_repairs=[bad])
        bad = dict(repair)
        blob = bytearray(bad["payload"]["raw"])
        off = bad["payload"]["vertoff"] + base["nv"] * 56 + 24
        struct.pack_into("<I", blob, off, 3)
        bad["payload"] = m.decode(bytes(blob))
        with self.assertRaisesRegex(m.RebindError, "compact binding IDs"):
            m.merge_source_topology_repairs(base, duplicate_repairs=[bad])

    def test_duplicate_and_edge_refinements_compose_only_on_disjoint_parents(self):
        base = fixture()
        # A separate synthetic source with disconnected faces lets edge split
        # and duplicate-only replacement prove the mixed dispatch contract.
        base = make_disconnected_fixture()
        edge = make_edge_repair(base, 0, 1)
        dup = duplicate_patch(base, 1, 3)
        raw, report = m.merge_source_topology_repairs(base, edge_repairs=[edge], duplicate_repairs=[dup])
        out = m.decode(raw)
        self.assertEqual(out["nv"], base["nv"] + 2)
        self.assertEqual(report["added_duplicate_vertices"], 1)
        self.assertEqual(len(report["face_parent_source_rows"]), len(out["inds"]))
        with self.assertRaisesRegex(m.RebindError, "overlapping regional"):
            m.merge_source_topology_repairs(base, edge_repairs=[edge], duplicate_repairs=[duplicate_patch(base, 0, 0)])


def make_disconnected_fixture():
    base = fixture()
    n = base["nv"]
    verts = base["verts"].copy()
    verts[3, :3] += np.asarray([3.0, 0.0, 0.0], dtype=np.float32)
    verts[2, :3] += np.asarray([3.0, 0.0, 0.0], dtype=np.float32)
    # Rewrite compact rows and dense data are unchanged; geometry is independent.
    raw = bytearray(base["raw"])
    for row in (2, 3):
        struct.pack_into("<3f", raw, base["vertoff"] + row * 56, *verts[row, :3])
    raw = bytearray(raw)
    # Face 1 becomes a disjoint triangle with vertices 3,4,2; move row 4 too.
    struct.pack_into("<3f", raw, base["vertoff"] + 4 * 56, 4.0, 1.0, 0.0)
    struct.pack_into("<3I", raw, base["indexoff"] + 12, 3, 4, 2)
    return m.decode(bytes(raw))


def make_edge_repair(base, a, b):
    n = base["nv"]
    vertices = np.vstack((base["verts"], base["verts"][a].copy()))
    vertices[n, :3] = (base["verts"][a, :3] + base["verts"][b, :3]) / 2
    weights = np.vstack((base["weights"], (base["weights"][a] + base["weights"][b]) / 2))
    faces, parents = [], []
    for parent, tri in enumerate(base["inds"]):
        tri = list(map(int, tri))
        for k in range(3):
            if {tri[k], tri[(k + 1) % 3]} == {a, b}:
                u, v, w = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
                faces.extend(((u, n, w), (n, v, w)))
                parents.extend((parent, parent))
                break
        else:
            faces.append(tuple(tri)); parents.append(parent)
    header = bytearray(base["raw"][:60])
    struct.pack_into("<2I", header, 16, n + 1, len(faces) * 3)
    raw = (bytes(header) + base["binds"].tobytes() + vertices.astype("<f4").tobytes() +
           np.asarray(faces, dtype="<u4").tobytes() + weights.astype("<f4").tobytes())
    return m.decode(raw), np.asarray(parents), np.asarray([[a, b]]), np.asarray([[.5, .5]])


if __name__ == "__main__":
    unittest.main()
