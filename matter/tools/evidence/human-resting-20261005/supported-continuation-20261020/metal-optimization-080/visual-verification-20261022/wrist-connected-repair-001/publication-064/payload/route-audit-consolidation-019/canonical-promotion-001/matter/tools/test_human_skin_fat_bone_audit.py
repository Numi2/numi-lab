import csv
import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np

MODULE = pathlib.Path(__file__).with_name("human_skin_fat_bone_audit.py")
spec = importlib.util.spec_from_file_location("route_audit_consolidation_019", MODULE)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


class RouteAuditConsolidationTests(unittest.TestCase):
    def test_unknown_endpoints_are_not_queried_and_never_create_planes(self):
        self.assertEqual(mod.geometry_only_endpoint_classification(None, 42, 20, 26), "not_queried")
        self.assertEqual(mod.geometry_only_endpoint_classification({}, 42, 20, 26), "outside_AABB_proven")
        # Geometry-only report policy is unconditional: endpoint status cannot
        # cause plane generation because this owner has no plane construction path.
        report_plane_fields = mod.geometry_only_plane_fields()
        self.assertEqual(report_plane_fields["construction_planes"], 0)
        self.assertEqual(report_plane_fields["construction_planes_world"], [])

    def test_query_prefers_native_emitted_candidate_nodes(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            initial = root / "initial-skin-fem-nodes.csv"
            exact = root / "exact-emitted.nodes.csv"
            initial.write_text("stale baseline")
            exact.write_text("candidate override")
            self.assertEqual(mod.matched_query_node_path(root), exact)
            exact.unlink()
            self.assertEqual(mod.matched_query_node_path(root), initial)

    def test_stale_initial_export_is_rejected_and_exact_emitted_export_passes(self):
        ids = np.array([10, 11, 12], dtype=np.int64)
        candidate = np.array([[1.25, 2.5, 3.75], [4.0, 5.0, 6.0], [-1.0, 0.25, 2.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            exact = root / "exact-emitted.nodes.csv"
            stale = root / "initial-skin-fem-nodes.csv"
            header = "global_node,layer,row,x_m,y_m,z_m\n"
            exact.write_text(header + "".join(
                f"{gid},0,{i},{x},{y},{z}\n"
                for i, (gid, (x, y, z)) in enumerate(zip(ids, candidate))))
            stale_points = candidate.copy()
            stale_points[0, 0] -= 0.001
            stale.write_text(header + "".join(
                f"{gid},0,{i},{x},{y},{z}\n"
                for i, (gid, (x, y, z)) in enumerate(zip(ids, stale_points))))

            with mod.matched_query_node_path(root, exact.name).open(newline="") as f:
                exact_rows = list(csv.DictReader(f))
            mod.validate_node_query_parity(ids, candidate, exact_rows)
            with mod.matched_query_node_path(root, stale.name).open(newline="") as f:
                stale_rows = list(csv.DictReader(f))
            with self.assertRaisesRegex(ValueError, "world positions differ"):
                mod.validate_node_query_parity(ids, candidate, stale_rows)
            with self.assertRaisesRegex(ValueError, "matched query node file must be"):
                mod.matched_query_node_path(root, "untrusted.csv")

    def test_unique_tet_edges_are_sorted_and_deduplicated(self):
        tets = np.array([[0, 1, 2, 3], [3, 2, 1, 4]], dtype=np.int64)
        edges = mod.unique_undirected_tet_edges(tets)
        expected = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3],
                             [1, 4], [2, 3], [2, 4], [3, 4]], dtype=np.int64)
        self.assertTrue(np.array_equal(edges, expected))
        self.assertTrue(np.array_equal(edges, mod.unique_undirected_tet_edges(tets[::-1])))
        self.assertEqual(len({tuple(row) for row in edges.tolist()}), len(edges))

    def test_outside_segment_endpoints_can_cross_two_bank_triangles(self):
        exact = mod.runpy.run_path(str(MODULE.parent.parent / "root_exact_segment_witness.py"))
        # Closed unit right tetrahedron: the segment starts outside, crosses x=0
        # and x+y+z=1 faces, then ends outside again.
        a = np.array([-1.0, 0.1, 0.1])
        b = np.array([2.0, 0.1, 0.1])
        faces = np.array([
            [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
            [[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            [[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]],
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        ])
        stable = np.array([10, 11, 12, 13], dtype=np.int64)
        result = mod.exact_segment_triangle_hits(a, b, faces, stable,
                                                exact["classify_segment_triangle"])
        self.assertEqual([h["classification"] for h in result["hits"]],
                         ["proper_interior_hit", "proper_interior_hit"])
        self.assertEqual([h["triangle_stable_id"] for h in result["hits"]], [11, 13])
        self.assertAlmostEqual(result["hits"][0]["exact_result"]["t"], 1/3)
        self.assertAlmostEqual(result["hits"][1]["exact_result"]["t"], 0.6)
        # Both endpoints lie outside the tetrahedron because their x-coordinate
        # is negative or their coordinate sum exceeds one.
        self.assertLess(a[0], 0.0)
        self.assertGreater(b.sum(), 1.0)
        candidate_points = np.array([a, b, [0., -2., 0.], [0., 0., 2.]])
        candidate_tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
        typed_ids = np.array([[7, 9, 3, stable_id] for stable_id in stable], dtype=np.int64)
        edge_report = mod.audit_unique_tet_edges(
            np.array([100, 101, 102, 103]), candidate_tets, typed_ids, faces,
            {7: candidate_points}, exact["classify_segment_triangle"])
        through_edge = [r for r in edge_report["witnesses"] if r["endpoint_local_nodes"] == [0, 1]]
        self.assertEqual([r["triangle_stable_id"] for r in through_edge], [11, 13])
        self.assertEqual(edge_report["counts"]["unique_tetrahedral_edges"], 6)

    def test_bank_local_hit_triangle_reconstructs_reported_segment_hit(self):
        # The hit index is local to a typed bank, not to the 469k-row global table.
        tri0 = np.array([[50, 50, 50], [51, 50, 50], [50, 51, 50]], dtype=np.float64)
        tri1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        bank = np.stack([tri0, tri1])
        stable_ids = np.array([9001, 295], dtype=np.int64)
        tri = mod.bank_local_hit_triangle(bank, stable_ids, 1, 295)
        a = np.array([0.25, 0.25, -1.0])
        b = np.array([0.25, 0.25, 1.0])
        # Reconstruct from the actual exact segment/triangle classifier output.
        exact = mod.runpy.run_path(str(MODULE.parent.parent / "root_exact_segment_witness.py"))
        result = exact["classify_segment_triangle"](a, b, tri)
        self.assertEqual(result["classification"], "proper_interior_hit")
        err = mod.validate_hit_reconstruction(tri, np.asarray(result["barycentric"]), a, b, result["t"])
        self.assertLess(err, 1e-15)
        with self.assertRaisesRegex(ValueError, "stable triangle ID disagree"):
            mod.bank_local_hit_triangle(bank, stable_ids, 1, 9001)
        with self.assertRaisesRegex(IndexError, "out of range"):
            mod.bank_local_hit_triangle(bank, stable_ids, 2, 0)

    def test_start_side_normal_is_not_mislabeled_exterior_for_inside_start(self):
        tri = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        matrix = np.eye(3)
        start = np.array([0.25, 0.25, 1.0])
        end = np.array([0.25, 0.25, -1.0])
        hit = np.array([0.25, 0.25, 0.0])
        start_side, exterior, basis = mod.orient_hit_normal(
            tri, matrix, start, end, hit, "inside", "outside")
        self.assertIsNotNone(exterior)
        self.assertEqual(basis, "exterior_toward_outside_end")
        self.assertGreater(float(np.dot(start_side, np.array([0., 0., 1.]))), 0.0)
        self.assertLess(float(np.dot(exterior, np.array([0., 0., 1.]))), 0.0)
        _, unknown_exterior, unknown_basis = mod.orient_hit_normal(
            tri, matrix, start, end, hit, "inside", "not_queried")
        self.assertIsNone(unknown_exterior)
        self.assertEqual(unknown_basis, "inside_start_end_unclassified_exterior_unknown")

    def test_external_ids_require_exact_topology_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            points = np.zeros((mod.EXPECTED_NODES, 3), dtype=np.float32)
            gids = np.arange(10000, 10000 + mod.EXPECTED_NODES, dtype=np.int64)
            tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
            mats = np.array([2], dtype=np.int8)
            candidate = root / "candidate.npz"
            ids = root / "ids.npz"
            np.savez(candidate, points_world_f32=points, tetrahedra=tets, material_ids=mats)
            np.savez(ids, global_nodes=gids, tetrahedra=tets, material_ids=mats)
            got_points, got_ids, pins = mod.load_candidate_points(
                candidate, mod.sha256(candidate), ids, mod.sha256(ids))
            self.assertTrue(np.array_equal(got_points, points))
            self.assertTrue(np.array_equal(got_ids, gids))
            self.assertIn("row identity verified", pins["candidate_id_source"])

            np.savez(ids, global_nodes=gids, tetrahedra=np.array([[0, 1, 2, 4]]), material_ids=mats)
            with self.assertRaisesRegex(ValueError, "row mapping is unproven"):
                mod.load_candidate_points(candidate, mod.sha256(candidate), ids, mod.sha256(ids))

    def test_matched_query_position_mismatch_fails(self):
        ids = np.array([10, 11, 12], dtype=np.int64)
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        rows = [
            dict(global_node=str(g), x_m=str(x), y_m=str(y), z_m=str(z), layer=str(i // 2411), row=str(i % 2411))
            for i, (g, (x, y, z)) in enumerate(zip(ids, points))
        ]
        rows[1]["x_m"] = "1.0001"
        with self.assertRaisesRegex(ValueError, "world positions differ"):
            mod.validate_node_query_parity(ids, points, rows)

    def test_matched_query_id_mismatch_fails(self):
        ids = np.array([10, 11, 12], dtype=np.int64)
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        rows = [
            dict(global_node=str(g), x_m=str(x), y_m=str(y), z_m=str(z), layer=str(i // 2411), row=str(i % 2411))
            for i, (g, (x, y, z)) in enumerate(zip(ids, points))
        ]
        rows[1]["global_node"] = "99"
        with self.assertRaisesRegex(ValueError, "global-node sequence differs"):
            mod.validate_node_query_parity(ids, points, rows)

    def test_matched_query_body_frame_mismatch_fails(self):
        bank = {7: {"position_x_m": "1", "position_y_m": "2", "position_z_m": "3",
                    "orientation_qx": "0", "orientation_qy": "0", "orientation_qz": "0", "orientation_qw": "1"}}
        poses = {7: {"current": {"position_world_m": [1, 2, 3],
                                 "orientation_world_xyzw": [0, 0, 0, 1]}}}
        mod.validate_body_pose_parity(bank, poses, np.array([7]))
        poses[7]["current"]["position_world_m"] = [1.01, 2, 3]
        with self.assertRaisesRegex(ValueError, "position frame differs"):
            mod.validate_body_pose_parity(bank, poses, np.array([7]))


if __name__ == "__main__":
    unittest.main()

