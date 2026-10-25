import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE_PATH = Path(__file__).with_name("same_parent_flips_358.py")
spec = importlib.util.spec_from_file_location("same_parent_flips_358", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def fixture(two_patches=False):
    xy = [
        (0.00, 0.00), (0.40, 0.00), (0.40, 0.20), (0.00, 0.60)
    ]
    faces = [[1, 3, 0], [3, 1, 2]]
    if two_patches:
        xy += [
            (0.50, 0.00), (0.66, 0.00), (0.66, 0.08), (0.50, 0.24)
        ]
        faces += [[5, 7, 4], [7, 5, 6]]
    points = np.asarray([(x, y, 0.0) for x, y in xy], dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    ref = np.asarray([(x, y, 0.0) for x, y in xy], dtype=np.float64)
    bary_nodes = {
        i: np.asarray([1.0 - x - y, x, y], dtype=np.float64)
        for i, (x, y) in enumerate(xy)
    }
    bary = np.asarray([[bary_nodes[int(n)] for n in tri] for tri in f])
    nface = len(f)
    nsource = 2 if not two_patches else 4
    return {
        "points_world_m": points,
        "source_O_faces": f.copy(),
        "area_reference_037_points_world_m_f64": ref.copy(),
        "area_reference_037_source_O_faces": f.copy(),
        "refinement_face_parent_face_row": np.full(nface, 7, dtype=np.int64),
        "refinement_face_parent_barycentric": bary,
        "face_kind": np.ones(nface, dtype=np.int8),
        "parent_coarse_face_row": np.full(nface, 3, dtype=np.int32),
        "parent_fine_detail_face_row": np.arange(10, 10 + nface, dtype=np.int32),
        "parent_source_face_marker": np.arange(30, 30 + nface, dtype=np.int32),
        "parent_fine_subface_row": np.arange(50, 50 + nface, dtype=np.int32),
        "face_source_ancestry_indptr": np.arange(nface + 1, dtype=np.int64),
        "face_source_ancestry_face_rows": np.arange(100, 100 + nface, dtype=np.int32),
        "face_fine_subface_ancestry_indptr": np.arange(nface + 1, dtype=np.int64),
        "face_fine_subface_ancestry_rows": np.arange(200, 200 + nface, dtype=np.int32),
        "coarse_boundary_anchor_rows": np.asarray([0], dtype=np.int32),
        "coarse_boundary_anchor_source_nodes": np.asarray([123], dtype=np.int32),
        "node_kind": np.arange(len(points), dtype=np.int8),
    }


class SameParentFlipTests(unittest.TestCase):
    def test_targeted_flip_updates_both_face_domains_barycentrics_and_ancestry(self):
        source = fixture()
        points_before = source["points_world_m"].copy()
        anchors_before = source["coarse_boundary_anchor_rows"].copy()
        out, report = module.propose_flip_batch(source, [0])
        self.assertEqual(report["accepted_pair_count"], 1)
        self.assertEqual(report["changed_face_rows"], [0, 1])
        self.assertTrue(np.array_equal(out["source_O_faces"], out["area_reference_037_source_O_faces"]))
        self.assertTrue(np.array_equal(out["points_world_m"], points_before))
        self.assertTrue(np.array_equal(out["coarse_boundary_anchor_rows"], anchors_before))
        self.assertEqual(report["boundary_edge_count_before"], report["boundary_edge_count_after"])
        self.assertTrue(report["immediate_parent_boundary_rings_preserved"])
        self.assertEqual(report["operations"][0]["parent_row"], 7)
        self.assertGreater(report["operations"][0]["angle_gain_deg"], 0.0)
        self.assertLess(report["operations"][0]["reference_parent_area_vector_delta_m2"], 1.0e-14)
        self.assertTrue(np.array_equal(out["face_source_ancestry_face_rows"], np.asarray([100, 101, 100, 101], dtype=np.int32)))
        self.assertTrue(np.array_equal(out["face_fine_subface_ancestry_rows"], np.asarray([200, 201, 200, 201], dtype=np.int32)))
        self.assertTrue(np.all(out["parent_fine_detail_face_row"] == -1))
        bary_by_node = {
            int(key): np.asarray(value)
            for key, value in report["operations"][0]["barycentric_by_vertex"].items()
        }
        for row in report["changed_face_rows"]:
            for k, node in enumerate(out["source_O_faces"][row]):
                self.assertTrue(np.allclose(out["refinement_face_parent_barycentric"][row, k], bary_by_node[int(node)], atol=0, rtol=0))
        self.assertFalse(report["geometry_admitted"])
        self.assertTrue(report["exact_full_surface_self_audit_required"])

    def test_batch_is_deterministic_and_vertex_disjoint(self):
        source = fixture(two_patches=True)
        a, ra = module.propose_flip_batch(source, [0, 2], max_flips=2)
        b, rb = module.propose_flip_batch(source, [2, 0], max_flips=2)
        self.assertEqual(ra["accepted_pair_count"], 2)
        self.assertEqual(ra["operations"], rb["operations"])
        self.assertTrue(np.array_equal(a["source_O_faces"], b["source_O_faces"]))
        self.assertEqual(len(set(ra["changed_face_rows"])), 4)

    def test_cross_parent_candidate_is_rejected(self):
        source = fixture()
        source["refinement_face_parent_face_row"][1] = 8
        out, report = module.propose_flip_batch(source, [0])
        self.assertEqual(report["accepted_pair_count"], 0)
        self.assertIn("parent_mismatch", report["rejection_reason_counts"])
        self.assertTrue(np.array_equal(out["source_O_faces"], source["source_O_faces"]))

    def test_current_reference_orientation_disagreement_is_rejected(self):
        source = fixture()
        source["area_reference_037_points_world_m_f64"][:, 1] *= -1.0
        out, report = module.propose_flip_batch(source, [0])
        self.assertEqual(report["accepted_pair_count"], 0)
        self.assertIn("current_reference_orientation_disagreement", report["rejection_reason_counts"])

    def test_missing_source_ancestry_is_rejected(self):
        source = fixture()
        source["face_source_ancestry_indptr"] = np.asarray([0, 0, 1], dtype=np.int64)
        source["face_source_ancestry_face_rows"] = np.asarray([101], dtype=np.int32)
        _, report = module.propose_flip_batch(source, [0])
        self.assertEqual(report["accepted_pair_count"], 0)
        self.assertIn("missing_face_ancestry", report["rejection_reason_counts"])

    def test_face_kind_mismatch_is_rejected(self):
        source = fixture()
        source["face_kind"][1] = 2
        _, report = module.propose_flip_batch(source, [0])
        self.assertEqual(report["accepted_pair_count"], 0)
        self.assertIn("face_kind_mismatch", report["rejection_reason_counts"])

    def test_unknown_face_aligned_payload_fails_closed(self):
        source = fixture()
        source["unclassified_face_payload"] = np.zeros((2, 5), dtype=np.float64)
        with self.assertRaises(module.FlipInputError):
            module.propose_flip_batch(source, [0])

    def test_reference_connectivity_mismatch_fails_closed(self):
        source = fixture()
        source["area_reference_037_source_O_faces"][0] = [0, 1, 2]
        with self.assertRaises(module.FlipInputError):
            module.propose_flip_batch(source, [0])

    def test_nonconvex_parent_patch_is_rejected(self):
        source = fixture()
        # Move the last polygon vertex into the concave side in both domains,
        # and keep its parent weights/ref coordinates consistent.
        source["points_world_m"][3, :2] = [0.10, 0.05]
        source["area_reference_037_points_world_m_f64"][3, :2] = [0.10, 0.05]
        source["refinement_face_parent_face_row"][0:2] = 7
        bary = source["refinement_face_parent_barycentric"]
        xy = source["area_reference_037_points_world_m_f64"][:, :2]
        nodes = {i: np.asarray([1.0-x-y, x, y]) for i,(x,y) in enumerate(xy)}
        source["refinement_face_parent_barycentric"] = np.asarray(
            [[nodes[int(n)] for n in tri] for tri in source["source_O_faces"]]
        )
        _, report = module.propose_flip_batch(source, [0])
        self.assertEqual(report["accepted_pair_count"], 0)
        self.assertIn("parent_patch_nonconvex_or_degenerate", report["rejection_reason_counts"])

    def test_allowed_edge_whitelist_cannot_admit_other_seed_edge(self):
        source = fixture(two_patches=True)
        _, original = module.propose_flip_batch(source, [0, 2], max_flips=2)
        self.assertEqual(original["accepted_pair_count"], 2)
        only_edge = [original["operations"][0]["edge"]]
        out, report = module.propose_flip_batch(
            source, [0, 2], max_flips=2, allowed_edges=only_edge
        )
        self.assertEqual(report["accepted_pair_count"], 1)
        self.assertEqual(report["operations"][0]["edge"], only_edge[0])
        self.assertTrue(report["immediate_parent_boundary_rings_preserved"])

    def test_excluded_seed_operation_is_removed_without_refill(self):
        source = fixture(two_patches=True)
        _, original = module.propose_flip_batch(source, [0, 2], max_flips=2)
        self.assertEqual(original["accepted_pair_count"], 2)
        removed = original["operations"][0]
        allowed = [op["edge"] for op in original["operations"]]
        blacklist = removed["face_rows"]
        out, report = module.propose_flip_batch(
            source, [0, 2], max_flips=2,
            allowed_edges=allowed, excluded_face_rows=blacklist
        )
        self.assertEqual(report["accepted_pair_count"], 1)
        self.assertNotIn(removed["edge"], [op["edge"] for op in report["operations"]])
        self.assertFalse(set(report["changed_face_rows"]) & set(blacklist))
        self.assertTrue(set(tuple(op["edge"]) for op in report["operations"]) <= set(tuple(sorted(e)) for e in allowed))
        self.assertTrue(report["immediate_parent_boundary_rings_preserved"])

    def test_seed_filter_validation_fails_closed(self):
        source = fixture()
        with self.assertRaises(module.FlipInputError):
            module.propose_flip_batch(source, [0], allowed_edges=[[0, 999]])
        with self.assertRaises(module.FlipInputError):
            module.propose_flip_batch(source, [0], excluded_face_rows=[True])
        with self.assertRaises(module.FlipInputError):
            module.propose_flip_batch(source, [0], allowed_edges=[[0, 1], [1, 0]])

    def test_zero_flip_budget_keeps_topology_unchanged(self):
        source = fixture()
        out, report = module.propose_flip_batch(source, [0], max_flips=0)
        self.assertEqual(report["accepted_pair_count"], 0)
        self.assertTrue(np.array_equal(out["source_O_faces"], source["source_O_faces"]))

    def test_empty_targets_and_out_of_range_fail(self):
        source = fixture()
        with self.assertRaises(module.FlipInputError):
            module.propose_flip_batch(source, [])
        with self.assertRaises(module.FlipInputError):
            module.propose_flip_batch(source, [2])


if __name__ == "__main__":
    unittest.main(verbosity=2)

