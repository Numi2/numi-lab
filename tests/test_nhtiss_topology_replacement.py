#!/usr/bin/env python3
import importlib.util
import struct
import unittest
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "tools" / "rebind_human_tissue_reference.py"
spec = importlib.util.spec_from_file_location("rebind_human_tissue_reference_topology_test", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _fixture():
    records = np.asarray([
        [0, 2, 0, 5, 0, 6, 134, 1],
        [2, 2, 5, 3, 6, 3, 999, 2],
    ], dtype="<u4")
    bindings = np.zeros(4, dtype=module.BIND_DTYPE)
    for index, body in enumerate((10, 11, 20, 21)):
        bindings[index]["body"] = body
        bindings[index]["transform"] = np.asarray([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32)
    vertices = np.zeros(8, dtype=module.VERT_DTYPE)
    points = np.asarray([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0],
        [2, 0, 1], [3, 0, 1], [2, 1, 1],
    ], dtype=np.float32)
    vertices["p"] = points
    vertices["n"] = np.asarray([0, 0, 1], dtype=np.float32)
    vertices["binding"] = 0
    vertices["w"][:, 0] = 1
    indices = np.asarray([0, 1, 2, 0, 2, 3, 5, 6, 7], dtype="<u4")
    registration = 77
    source_id = bytes(range(32))
    header = module.HEADER.pack(b"NHTISS4\0", 5, len(records), len(bindings), len(vertices),
                               len(indices), registration, source_id)
    raw = header + records.tobytes() + bindings.tobytes() + vertices.tobytes() + indices.tobytes()
    return raw, records, bindings, vertices, indices


class NHTISSTopologyReplacementTests(unittest.TestCase):
    def test_replaces_oriented_disk_and_preserves_unrelated_payload(self):
        raw, source_records, source_bindings, source_vertices, source_indices = _fixture()
        result, report = module.replace_surface_topology_patch(
            raw,
            stable_id=134,
            removed_face_rows=np.asarray([0, 1], dtype=np.int32),
            replacement_faces_local=np.asarray([
                [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
            ], dtype=np.int32),
            boundary_cycle_local=np.asarray([0, 1, 2, 3], dtype=np.int32),
            ancestry_offsets=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
            ancestry_face_rows=np.asarray([0, 0, 1, 1], dtype=np.int32),
            construction_ancestry_label="construction ancestry from the two replaced source faces",
        )
        parsed = module.decode_nhtiss(result)
        self.assertEqual(parsed["nr"], 2)
        self.assertEqual(parsed["nb"], 4)
        self.assertEqual(parsed["nv"], 8)
        self.assertEqual(parsed["ni"], 15)
        self.assertEqual(int(parsed["records"][0, 5]), 12)
        self.assertEqual(int(parsed["records"][1, 4]), 12)
        np.testing.assert_array_equal(parsed["records"][:, [0, 1, 2, 3, 6, 7]],
                                      source_records[:, [0, 1, 2, 3, 6, 7]])
        np.testing.assert_array_equal(parsed["bindings"], source_bindings)
        np.testing.assert_array_equal(parsed["vertices"], source_vertices)
        np.testing.assert_array_equal(parsed["indices"][:12],
                                      np.asarray([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype="<u4"))
        np.testing.assert_array_equal(parsed["indices"][12:], source_indices[6:])
        self.assertEqual(report["source_face_row_to_candidate_face_row"], [None, None])
        self.assertEqual(report["new_candidate_face_rows"], [0, 1, 2, 3])
        self.assertTrue(report["preservation"]["all_vertex_position_normal_and_route_weight_bytes_exact"])
        self.assertFalse(report["mechanical_face_map"]["previous_map_valid"])
        self.assertFalse(report["gate"]["physical_admission"])

    def test_rejects_boundary_orientation_and_missing_ancestry(self):
        raw, *_ = _fixture()
        common = dict(
            stable_id=134,
            removed_face_rows=np.asarray([0, 1], dtype=np.int32),
            replacement_faces_local=np.asarray([
                [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
            ], dtype=np.int32),
            ancestry_offsets=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
            ancestry_face_rows=np.asarray([0, 0, 1, 1], dtype=np.int32),
            construction_ancestry_label="construction lineage only",
        )
        with self.assertRaises(module.ReferenceError):
            module.replace_surface_topology_patch(
                raw, boundary_cycle_local=np.asarray([0, 3, 2, 1], dtype=np.int32), **common)
        with self.assertRaises(module.ReferenceError):
            module.replace_surface_topology_patch(
                raw, boundary_cycle_local=np.asarray([0, 1, 2, 3], dtype=np.int32),
                ancestry_offsets=np.asarray([0, 0, 2, 3, 4], dtype=np.int32), **{
                    key: value for key, value in common.items() if key != "ancestry_offsets"})

    def test_rejects_non_disk_or_out_of_range_source_face(self):
        raw, *_ = _fixture()
        with self.assertRaises(module.ReferenceError):
            module.replace_surface_topology_patch(
                raw, stable_id=134,
                removed_face_rows=np.asarray([0, 99], dtype=np.int32),
                replacement_faces_local=np.asarray([[0, 1, 4]], dtype=np.int32),
                boundary_cycle_local=np.asarray([0, 1, 2, 3], dtype=np.int32),
                ancestry_offsets=np.asarray([0, 1], dtype=np.int32),
                ancestry_face_rows=np.asarray([0], dtype=np.int32),
                construction_ancestry_label="construction ancestry",
            )


if __name__ == "__main__":
    unittest.main()
