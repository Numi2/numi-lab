import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from validate_stream_checkpoint_009 import ValidationError, _compact_capacity_rows, validate_parts_and_prefix


class CheckpointValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.parts = Path(self.temp.name)
        self.points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
        self.tets = np.asarray([[0, 1, 2, 3]], np.int64)
        self.fixed = np.asarray([True, False, False, False])
        self.inventory = [{"key": "bank_0000", "triangles": 2, "kind": "soft_tissue",
                           "stable_id": 7, "layer": 1, "body_indices": [2, 3]}]

    def tearDown(self):
        self.temp.cleanup()

    def state_for(self, *, ranges=((0, 2),), face_ids=(0,), bad=(0, 1, 2), zero=(2,)):
        # One polygon over face 0; face 1 is deliberately a processed zero-hit face.
        bary = np.asarray([[.8, .2, 0, 0], [.7, .3, 0, 0], [1, 0, 0, 0]], np.float64)
        tids = np.asarray([0], np.int64)
        faces = np.asarray(face_ids, np.int64)
        offsets = np.asarray([0, len(bary)], np.int64)
        np.savez(self.parts / "part-00000.npz", tet_ids=tids, face_ids=faces,
                 bank_ids=np.asarray([0], np.uint16), poly_offsets=offsets,
                 polygon_barycentric=bary,
                 insufficient_capacity_polygon_vertex_rows=np.asarray(bad, np.int64),
                 zero_capacity_polygon_vertex_rows=np.asarray(zero, np.int64))
        lo, hi = ranges[0]
        meta = {"file": "part-00000.npz", "sha256": "", "bank_id": 0,
                "bank_key": "bank_0000", "kind": "soft_tissue", "stable_id": 7,
                "layer": 1, "body_indices": [2, 3], "local_face_range": [lo, hi],
                "flat_face_range": [lo, hi], "faces_processed": hi-lo,
                "polygons": 1, "polygon_vertices": 3,
                "insufficient_capacity_rows": len(bad), "zero_capacity_rows": len(zero),
                "aabb_candidates": 1, "first_witnesses": []}
        return {"parts": [meta], "faces_processed": hi-lo, "polygons": 1,
                "insufficient_capacity_rows": len(bad), "zero_capacity_rows": len(zero),
                "next_bank": 1, "next_local_face": 0, "complete": True,
                "native_admission": False, "physical_admission": False}

    def test_zero_hit_faces_are_covered_by_interval_not_polygon_ids(self):
        state = self.state_for()
        report = validate_parts_and_prefix(state, self.inventory, self.tets, self.fixed,
                                           self.parts, points=self.points, verify_part_sha=False)
        self.assertEqual(report["processed_faces"], 2)
        self.assertEqual(report["distinct_hit_faces_in_prefix"], 1)
        self.assertEqual(report["zero_hit_faces_in_processed_prefix"], 1)

    def test_witness_world_point_is_recomputed(self):
        state = self.state_for()
        meta = state["parts"][0]
        meta["first_witnesses"] = [{"bank_id": 0, "kind": "soft_tissue", "stable_id": 7,
            "layer": 1, "local_face": 0, "tet_row": 0, "tet_nodes": [0, 1, 2, 3],
            "barycentric": [.8, .2, 0, 0], "capacity": .2,
            "world_m": [.2, 0, 0], "positive_support_on_fixed_only": False}]
        validate_parts_and_prefix(state, self.inventory, self.tets, self.fixed,
                                  self.parts, points=self.points, verify_part_sha=False)
        meta["first_witnesses"][0]["world_m"] = [99, 99, 99]
        with self.assertRaisesRegex(ValidationError, "world point mismatch"):
            validate_parts_and_prefix(state, self.inventory, self.tets, self.fixed,
                                      self.parts, points=self.points, verify_part_sha=False)

    def test_gap_and_overlap_in_face_ranges_rejected(self):
        state = self.state_for(ranges=((1, 2),))
        with self.assertRaisesRegex(ValidationError, "expected local 0"):
            validate_parts_and_prefix(state, self.inventory, self.tets, self.fixed, self.parts, verify_part_sha=False)

    def test_orphan_capacity_row_rejected(self):
        state = self.state_for(bad=(0, 9), zero=(2,))
        with self.assertRaisesRegex(ValidationError, "orphan insufficient-capacity"):
            validate_parts_and_prefix(state, self.inventory, self.tets, self.fixed, self.parts, verify_part_sha=False)

    def test_wrong_typed_identity_rejected(self):
        state = self.state_for()
        state["parts"][0]["stable_id"] = 8
        with self.assertRaisesRegex(ValidationError, "typed identity mismatch"):
            validate_parts_and_prefix(state, self.inventory, self.tets, self.fixed, self.parts, verify_part_sha=False)

    def test_compact_index_keeps_bank_local_flat_tet_and_support_identity(self):
        tets = np.asarray([[10, 11, 12, 13], [20, 21, 22, 23]], np.int64)
        bary = np.asarray([[0, .25, .75, 0], [0, .25, .75, 0], [.5, 0, .5, 0]], np.float64)
        idx = _compact_capacity_rows(4, np.asarray([7, 7, 9]), np.asarray([107, 107, 109]),
                                     np.asarray([1, 1, 0]), bary,
                                     np.asarray([0., 0., 0.]), np.ones(3, bool), tets)
        self.assertEqual(idx["keys"].tolist(), [[4, 7, 107, 1], [4, 9, 109, 0]])
        self.assertEqual(idx["raw_row_counts"].tolist(), [2, 1])
        self.assertEqual(idx["stored_row_counts"].tolist(), [1, 1])
        self.assertEqual(idx["support_node_ids"].tolist(), [[2**32-1, 21, 22, 2**32-1],
                                                              [10, 2**32-1, 12, 2**32-1]])


if __name__ == "__main__":
    unittest.main()
