from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import outward_ray_exit_098 as ray

PREDICATE_PATH = Path(
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/"
    "human-resting-20261005/supported-continuation-20261020/metal-optimization-080/"
    "visual-verification-20261022/wrist-connected-repair-001/route-audit-consolidation-019/"
    "canonical-promotion-001/matter/root_exact_segment_witness.py"
)
PREDICATE_SHA = "d6a2efe451e7743b659d74c274d1372a5ed2a765511a8671b2c1a2370b7d03a2"
assert hashlib.sha256(PREDICATE_PATH.read_bytes()).hexdigest() == PREDICATE_SHA
spec = importlib.util.spec_from_file_location("exact_segment_predicate_098_test", PREDICATE_PATH)
exact_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact_module)
classify = exact_module.classify_segment_triangle


def tetra_faces(origin=(0.0, 0.0, 0.0)):
    o = np.asarray(origin, dtype=np.float64)
    v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64) + o
    return np.stack([v[[1,2,3]], v[[0,2,1]], v[[0,1,3]], v[[0,3,2]]])


def certify(start, dlocal, triangles, *, M=None, start_status="inside", lo=None, hi=None, clearance=5e-5):
    M = np.eye(3) if M is None else np.asarray(M, dtype=np.float64)
    lo = triangles.min(axis=(0,1)) if lo is None else np.asarray(lo, dtype=np.float64)
    hi = triangles.max(axis=(0,1)) if hi is None else np.asarray(hi, dtype=np.float64)
    end, _ = ray.ray_endpoint_outside_aabb(start, dlocal, lo, hi)
    world_start = np.linalg.solve(M, start)
    return ray.certify_first_outward_exit(
        start_status=start_status, start_local=start, end_local=end,
        source_direction_local=dlocal, triangles_local=triangles,
        triangle_stable_ids=np.arange(len(triangles), dtype=np.int64),
        bank_aabb_min_local=lo, bank_aabb_max_local=hi,
        expected_bank_triangle_count=len(triangles),
        world_start=world_start, world_to_local=M, clearance_m=clearance,
        exact_classifier=classify, exact_classifier_sha256=PREDICATE_SHA,
    )


class OutwardRayExit098Tests(unittest.TestCase):
    def test_parent_barycentric_source_direction_and_consistency(self):
        # Two parents share the same field on their common source face.
        source_ids = np.arange(6, dtype=np.int64)
        out_rows = np.array([0,1,2,3,4,5,-1], dtype=np.int64)
        outer = np.array([[1,0,0],[0,1,0]], dtype=np.float64)
        deep = np.zeros((2,3), dtype=np.float64)
        tets = np.array([[0,2,4,1],[0,2,4,3]], dtype=np.int64)
        gen_rows = np.array([6,6], dtype=np.int64)
        gen_parent = np.array([0,1], dtype=np.int64)
        bary = np.array([[.2,.3,.5,0],[.2,.3,.5,0]], dtype=np.float64)
        field, report = ray.build_source_outward_field(
            source_global_node_ids=source_ids, output_source_input_row=out_rows,
            outer_points_by_column_m=outer, deep_points_by_column_m=deep,
            native_parent_tetrahedra=tets, generated_output_rows=gen_rows,
            generated_parent_tet_rows=gen_parent, generated_parent_barycentric=bary,
            column_count=2,
        )
        np.testing.assert_allclose(field[6], [1,0,0], atol=1e-14)
        self.assertEqual(report["generated_direction_disagreements"], [])

    def test_disagreeing_parent_directions_fail_without_averaging(self):
        source_ids = np.arange(6, dtype=np.int64)
        out_rows = np.array([0,1,2,3,4,5,-1], dtype=np.int64)
        outer = np.array([[1,0,0],[0,1,0]], dtype=np.float64)
        deep = np.zeros((2,3), dtype=np.float64)
        tets = np.array([[0,2,4,1],[1,3,5,2]], dtype=np.int64)
        rows = np.array([6,6], dtype=np.int64)
        parents = np.array([0,1], dtype=np.int64)
        bary = np.array([[.2,.3,.5,0],[.2,.3,.5,0]], dtype=np.float64)
        with self.assertRaises(ray.CertificationError):
            ray.build_source_outward_field(
                source_global_node_ids=source_ids, output_source_input_row=out_rows,
                outer_points_by_column_m=outer, deep_points_by_column_m=deep,
                native_parent_tetrahedra=tets, generated_output_rows=rows,
                generated_parent_tet_rows=parents, generated_parent_barycentric=bary,
                column_count=2,
            )
        field, report = ray.build_source_outward_field(
            source_global_node_ids=source_ids, output_source_input_row=out_rows,
            outer_points_by_column_m=outer, deep_points_by_column_m=deep,
            native_parent_tetrahedra=tets, generated_output_rows=rows,
            generated_parent_tet_rows=parents, generated_parent_barycentric=bary,
            column_count=2, strict_parent_agreement=False,
        )
        self.assertEqual(report["unresolved_output_rows"], [6])
        self.assertFalse(np.isfinite(field[6]).all())
        self.assertEqual(len(report["generated_direction_disagreements"]), 1)

    def test_global_ids_are_identity_only_and_parent_indices_are_input_rows(self):
        # IDs deliberately do not equal source-input rows or O/I/D indices.
        source_ids = np.array([8107, 8102, 8109, 8104, 8103, 8111], dtype=np.int64)
        out_rows = np.array([2, 0, 5, 1, 4, 3, -1], dtype=np.int64)
        outer = np.array([[1.,0,0],[0,1.,0]], dtype=np.float64)
        deep = np.zeros((2,3), dtype=np.float64)
        tets = np.array([[0,2,4,1]], dtype=np.int64)  # source-input rows, not IDs
        field, report = ray.build_source_outward_field(
            source_global_node_ids=source_ids, output_source_input_row=out_rows,
            outer_points_by_column_m=outer, deep_points_by_column_m=deep,
            native_parent_tetrahedra=tets,
            generated_output_rows=np.array([6], dtype=np.int64),
            generated_parent_tet_rows=np.array([0], dtype=np.int64),
            generated_parent_barycentric=np.array([[-1.6697754290362354e-13, .4, .3, .30000000000016696]]),
            column_count=2, strict_parent_agreement=True,
        )
        np.testing.assert_allclose(field[1], [1,0,0], atol=0, rtol=0) # direct input row 0
        np.testing.assert_allclose(field[0], [1,0,0], atol=0, rtol=0) # direct input row 2
        np.testing.assert_allclose(field[6], np.array([.7,.3,0])/np.sqrt(.58), atol=1e-12)
        self.assertLess(report["generated_parent_barycentric_min"], 0.0)
        self.assertLessEqual(report["generated_parent_barycentric_max_row_sum_error"], 1e-10)

    def test_sample_direction_uses_prolonged_edge_field(self):
        f = np.array([[1,0,0],[0,1,0]], dtype=np.float64)
        got = ray.interpolate_source_outward_direction(f, 0, 1, .25)
        np.testing.assert_allclose(got, np.array([.75,.25,0])/np.sqrt(.625), atol=1e-14)

    def test_first_exact_exit_from_inside_to_outside(self):
        triangles = tetra_faces()
        start = np.array([.1,.1,.1], dtype=np.float64)
        result = certify(start, np.array([1.,0,0]), triangles)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["endpoint_parity"], "odd")
        self.assertEqual(len(result["proper_hits"]), 1)
        hit = result["selected_first_exit"]
        self.assertEqual(hit["triangle_stable_id"], 0)
        self.assertAlmostEqual(hit["hit_local"][0], .8, places=14)
        self.assertAlmostEqual(sum(hit["barycentric"]), 1.0, places=14)
        self.assertGreater(np.dot(hit["unit_exit_normal_world"], [1,0,0]), 0)
        self.assertGreater(result["plane"]["minimum_signed_distance_m"], 0)

    def test_first_exit_selected_before_later_disconnected_component(self):
        triangles = np.concatenate([tetra_faces(), tetra_faces((1.5,0,0))])
        result = certify(np.array([.1,.1,.1]), np.array([1.,0,0]), triangles)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["proper_hits"]), 3)
        self.assertEqual(result["selected_first_exit"]["triangle_stable_id"], 0)
        tvalues = [float(h["t"]) for h in result["proper_hits"]]
        self.assertEqual(tvalues, sorted(tvalues))

    def test_plane_normal_uses_inverse_transpose_covector(self):
        triangles = tetra_faces()
        M = np.diag([2.,1.,1.])
        local_start = np.array([.1,.1,.1])
        result = certify(local_start, np.array([1.,0,0]), triangles, M=M)
        self.assertEqual(result["status"], "PASS")
        expected = np.array([2.,1.,1.]) / np.sqrt(6.0)
        np.testing.assert_allclose(result["plane"]["unit_normal_world"], expected, atol=1e-14)

    def test_even_crossing_parity_is_rejected(self):
        triangles = tetra_faces()
        start = np.array([.1,.1,.9]) # in bank AABB, outside the tetra interior
        result = certify(start, np.array([0.,0,-1.]), triangles, start_status="inside")
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(result["endpoint_parity"], "even")
        self.assertEqual(result["reason"], "inside_to_outside_ray_has_even_exact_hit_parity")

    def test_boundary_hit_fails_closed(self):
        triangles = tetra_faces()
        # Binary-rational endpoints make the segment hit the shared edge
        # (1/2, 1/2, 0) exactly; do not normalize/reconstruct this endpoint.
        start = np.array([.125, .125, .125], dtype=np.float64)
        end = np.array([1.25, 1.25, -.25], dtype=np.float64)
        direction = end - start
        direction /= np.linalg.norm(direction)
        lo = triangles.min(axis=(0, 1))
        hi = triangles.max(axis=(0, 1))
        result = ray.certify_first_outward_exit(
            start_status="inside", start_local=start, end_local=end,
            source_direction_local=direction, triangles_local=triangles,
            triangle_stable_ids=np.arange(len(triangles), dtype=np.int64),
            bank_aabb_min_local=lo, bank_aabb_max_local=hi,
            expected_bank_triangle_count=len(triangles),
            world_start=start, world_to_local=np.eye(3), clearance_m=5e-5,
            exact_classifier=classify, exact_classifier_sha256=PREDICATE_SHA,
        )
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(result["reason"], "boundary_coplanar_degenerate_or_tangent_witness")
        self.assertTrue(any(x["classification"] == "boundary_hit" for x in result["unresolved_witnesses"]))

    def test_duplicate_same_parameter_hits_fail_closed(self):
        triangles = tetra_faces()
        triangles = np.concatenate([triangles, triangles[[0]]])
        result = certify(np.array([.1,.1,.1]), np.array([1.,0,0]), triangles)
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(result["reason"], "nonunique_exact_hit_parameter")
        self.assertEqual(len(result["proper_hits"]), 2)

    def test_missing_intersections_and_wrong_status_reject(self):
        triangles = tetra_faces()
        result = certify(np.array([.9,.9,.9]), np.array([1.,0,0]), triangles)
        self.assertEqual(result["status"], "REJECT")
        self.assertEqual(result["reason"], "inside_to_outside_ray_has_no_exact_hits")
        result2 = certify(np.array([.1,.1,.1]), np.array([1.,0,0]), triangles, start_status="outside")
        self.assertEqual(result2["status"], "REJECT")
        self.assertEqual(result2["reason"], "ray_start_not_certified_inside")

    def test_aabb_must_cover_the_complete_supplied_bank(self):
        triangles = tetra_faces()
        with self.assertRaises(ray.CertificationError):
            certify(np.array([.1,.1,.1]), np.array([1.,0,0]), triangles,
                    lo=[0,0,0], hi=[.9,1,1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
