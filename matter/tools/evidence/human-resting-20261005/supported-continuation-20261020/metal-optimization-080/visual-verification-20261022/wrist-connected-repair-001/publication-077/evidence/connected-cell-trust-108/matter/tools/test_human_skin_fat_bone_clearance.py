import unittest
import tempfile
from pathlib import Path
import numpy as np
from human_skin_fat_bone_clearance import (
    build_edge_span_midpoint_samples,
    validate_sample_query_rows,
    closest_point_barycentric,
    nearest_exit_contact,
    validate_sample_points_csv,
    certify_collapsed_endpoint_aliases,
)


class ClosestExit093Tests(unittest.TestCase):
    def setUp(self):
        self.points = np.zeros((4, 3), dtype=np.float32)
        self.points[2] = [0.0, 0.0, 0.0]
        self.points[0] = [2.0, 0.0, 0.0]
        self.witnesses = [
            {"edge_index": 9, "endpoint_local_nodes": [2, 0], "endpoint_query_rows": [0, 1],
             "body": 4, "bone_stable_id": 7, "triangle_stable_id": 51,
             "classification": "proper_interior_hit",
             "exact_result": {"classification": "proper_interior_hit", "t": 0.25, "barycentric": [0.3,0.3,0.4]}},
            {"edge_index": 9, "endpoint_local_nodes": [2, 0], "endpoint_query_rows": [0, 1],
             "body": 4, "bone_stable_id": 7, "triangle_stable_id": 52,
             "classification": "proper_interior_hit",
             "exact_result": {"classification": "proper_interior_hit", "t": 0.75, "barycentric": [0.3,0.3,0.4]}},
        ]

    def test_midpoint_sample_uses_separate_query_and_mechanics_domains(self):
        samples, report = build_edge_span_midpoint_samples(
            self.witnesses, [], {(0, 4, 7): "outside", (1, 4, 7): "outside"},
            np.array([2, 0], dtype=np.int64), self.points,
        )
        self.assertEqual(len(samples), 3)
        s = samples[1]
        self.assertEqual(s["sample_query_row"], 1)
        self.assertEqual(s["source_domain"], "mechanics_node_row")
        self.assertEqual((s["mechanics_node_a"], s["mechanics_node_b"]), (2, 0))
        self.assertAlmostEqual(s["t_mid"], 0.5)
        np.testing.assert_array_equal(np.asarray(s["point_world_f32"], np.float32), [1, 0, 0])
        self.assertEqual(report["endpoint_parity_contradiction_count"], 0)
        self.assertEqual([q["expected_inside_by_exact_parity"] for q in samples], [False, True, False])

    def test_query_domain_and_actual_outside_classification_are_preserved(self):
        samples, _ = build_edge_span_midpoint_samples(
            self.witnesses, [], {(0, 4, 7): "outside", (1, 4, 7): "outside"},
            np.array([2, 0], dtype=np.int64), self.points,
        )
        points = np.asarray([s["point_world_f32"] for s in samples], np.float32)
        nodes = [
            {"query_row": str(i), "total_bone_banks": "185", "aabb_proven_outside_banks": "184",
             "parity_outside_banks": "0", "inside_banks": "1", "ambiguous_banks": "0", "invalid_banks": "0"}
            for i in range(3)
        ]
        pairs = [
            {"query_row": "0", "bank_body": "4", "bank_stable_id": "7", "classification": "outside", "ambiguous": "0"},
            {"query_row": "1", "bank_body": "4", "bank_stable_id": "7", "classification": "inside", "ambiguous": "0"},
            {"query_row": "2", "bank_body": "4", "bank_stable_id": "7", "classification": "outside", "ambiguous": "0"},
        ]
        qmap, status = validate_sample_query_rows(samples, points, nodes, pairs, aabb_candidate_keys={(i,4,7) for i in range(3)})
        self.assertEqual([status[i] for i in range(3)], ["outside", "inside", "outside"])
        self.assertEqual(qmap[(1,4,7)]["classification"], "inside")
        # An absent target pair is a valid outside result only if that bank was
        # proven outside the sample AABB; an AABB candidate missing its row fails.
        _, status = validate_sample_query_rows(samples, points, nodes, pairs[:1] + pairs[2:], aabb_candidate_keys={(0,4,7),(2,4,7)})
        self.assertEqual(status[1], "outside_aabb_proven")
        with self.assertRaisesRegex(ValueError, "AABB candidate target bank query is missing"):
            validate_sample_query_rows(samples, points, nodes, pairs[:1] + pairs[2:], aabb_candidate_keys={(0,4,7),(1,4,7),(2,4,7)})

    def test_parity_contradiction_and_uncertain_hits_reject(self):
        _, report = build_edge_span_midpoint_samples(
            self.witnesses[:1], [], {(0, 4, 7): "outside", (1, 4, 7): "outside"},
            np.array([2, 0], dtype=np.int64), self.points,
        )
        self.assertEqual(report["endpoint_parity_contradiction_count"], 1)
        with self.assertRaisesRegex(ValueError, "uncertain hit"):
            build_edge_span_midpoint_samples(
                self.witnesses, [{"edge_index": 9}], {(0, 4, 7): "outside", (1, 4, 7): "outside"},
                np.array([2, 0], dtype=np.int64), self.points,
            )

    def test_barycentric_nearest_exit_plane_and_mechanics_support(self):
        tri = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        bary, residual = closest_point_barycentric([0.25, 0.25, 0.0], tri)
        np.testing.assert_allclose(bary, [0.5, 0.25, 0.25], atol=1e-12)
        self.assertEqual(residual, 0.0)
        sample = {"sample_query_row": 0, "edge_index": 9, "mechanics_node_a": 2,
                  "mechanics_node_b": 0, "t_mid": 0.5, "body": 4, "bone_stable_id": 7}
        pair = {"query_row": "0", "bank_body": "4", "bank_stable_id": "7", "classification": "inside",
                "ambiguous": "0", "nearest_bone": "7", "nearest_triangle": "51",
                "query_local_x": "0.25", "query_local_y": "0.25", "query_local_z": "-0.2",
                "closest_x": "0.25", "closest_y": "0.25", "closest_z": "0.0"}
        row = nearest_exit_contact(sample, pair, tri, np.eye(3), np.zeros(3), np.zeros(3), node_count=4)
        self.assertEqual(row["mechanics_node_rows"], [2, 0])
        np.testing.assert_allclose(row["nearest_exit_world_normal"], [0, 0, 1], atol=1e-12)
        self.assertAlmostEqual(row["target_m"], 5e-5)
        with self.assertRaisesRegex(ValueError, "invalid barycentric identity"):
            closest_point_barycentric([1.2, 0.2, 0.0], tri)

    def test_points_csv_preserves_sample_float32_bits(self):
        samples, _ = build_edge_span_midpoint_samples(
            self.witnesses, [], {(0, 4, 7): "outside", (1, 4, 7): "outside"},
            np.array([2, 0], dtype=np.int64), self.points,
        )
        p = np.asarray([s["point_world_f32"] for s in samples], dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "points.csv"
            lines=["query_row,x_m,y_m,z_m,x_bits_u32,y_bits_u32,z_bits_u32"]
            for i in range(len(p)):
                bits=p.view(np.uint32)[i]
                lines.append(f"{i},{p[i,0]:.9g},{p[i,1]:.9g},{p[i,2]:.9g},{bits[0]},{bits[1]},{bits[2]}")
            path.write_text("\n".join(lines)+"\n")
            result = validate_sample_points_csv(path, samples, p)
        self.assertEqual(result["rows"], 3)
        self.assertTrue(result["bitwise_sample_identity"])

    def test_float32_collapsed_tiny_span_is_retained_as_failure_diagnostic(self):
        pts = np.array([[1.0, 0., 0.], [1.000001, 0., 0.]], dtype=np.float32)
        ws = [
            {"edge_index": 1, "endpoint_local_nodes": [0, 1], "endpoint_query_rows": [0, 1],
             "body": 4, "bone_stable_id": 7, "triangle_stable_id": 1,
             "classification": "proper_interior_hit", "exact_result": {"classification": "proper_interior_hit", "t": 0.1000000000, "barycentric": [1/3,1/3,1/3]}},
            {"edge_index": 1, "endpoint_local_nodes": [0, 1], "endpoint_query_rows": [0, 1],
             "body": 4, "bone_stable_id": 7, "triangle_stable_id": 2,
             "classification": "proper_interior_hit", "exact_result": {"classification": "proper_interior_hit", "t": 0.1000000001, "barycentric": [1/3,1/3,1/3]}},
        ]
        samples, report = build_edge_span_midpoint_samples(ws, [], {(0,4,7): "outside", (1,4,7): "outside"}, np.array([0,1]), pts)
        self.assertEqual(len(samples), 3)
        self.assertTrue(samples[1]["f32_collapsed_span"])
        self.assertEqual(report["f32_collapsed_sample_query_rows"], [0, 1])


class CollapsedEndpointAlias107Tests(unittest.TestCase):
    def _sample(self, points, node_a, node_b, query_a, query_b, *, expected=True, query_row=0):
        endpoint_node = node_b
        bits = points[endpoint_node].view(np.uint32).copy()
        values = points[endpoint_node].copy()
        return {
            "sample_id": 0, "sample_query_row": query_row, "source_domain": "mechanics_node_row",
            "edge_index": 3, "mechanics_node_a": node_a, "mechanics_node_b": node_b,
            "endpoint_query_rows": [query_a, query_b], "body": 4, "bone_stable_id": 7,
            "expected_inside_by_exact_parity": expected, "point_world_f32": values.tolist(),
            "point_world_f32_bits": bits.tolist(), "f32_collapsed_span": True,
        }

    def _certify(self, sample, points, endpoint_classes, actual, exit_ids=(0,), **overrides):
        args = dict(
            samples=[sample], mechanics_points_f32=points,
            query_row_to_mechanics_node=np.arange(len(points), dtype=np.int64),
            endpoint_classification=endpoint_classes,
            sample_target_classification={0: actual},
            certified_inside_exit_sample_ids=exit_ids,
            complete_endpoint_query_185=True, complete_sample_query_185=True,
            endpoint_query_identity_valid=True, sample_query_identity_valid=True,
        )
        args.update(overrides)
        return certify_collapsed_endpoint_aliases(**args)

    def test_inside_endpoint_alias_requires_matching_parity_and_exit_receipt(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
        sample = self._sample(points, 0, 1, 0, 1)
        report = self._certify(sample, points, {(0,4,7): "outside", (1,4,7): "inside"}, "inside")
        self.assertEqual(report["certified_endpoint_alias_sample_ids"], [0])
        self.assertEqual(report["endpoint_alias_exception_count"], 1)

    def test_outside_aabb_endpoint_alias_is_explicitly_supported(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
        sample = self._sample(points, 0, 1, 0, 1, expected=False)
        sample["point_world_f32"] = points[0].tolist()
        sample["point_world_f32_bits"] = points[0].view(np.uint32).tolist()
        report = self._certify(sample, points, {(0,4,7): "outside_aabb_proven", (1,4,7): "inside"},
                               "outside_aabb_proven", exit_ids=())
        self.assertEqual(report["certified_endpoint_alias_sample_ids"], [0])

    def test_nonendpoint_collapse_and_missing_inside_exit_fail_closed(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
        sample = self._sample(points, 0, 1, 0, 1)
        sample["point_world_f32"] = [0.5, 0., 0.]
        sample["point_world_f32_bits"] = np.asarray(sample["point_world_f32"], np.float32).view(np.uint32).tolist()
        with self.assertRaisesRegex(ValueError, "uncertified collapsed"):
            self._certify(sample, points, {(0,4,7): "outside", (1,4,7): "inside"}, "inside")
        sample["point_world_f32"] = points[1].tolist()
        sample["point_world_f32_bits"] = points[1].view(np.uint32).tolist()
        with self.assertRaisesRegex(ValueError, "uncertified collapsed"):
            self._certify(sample, points, {(0,4,7): "outside", (1,4,7): "inside"}, "inside", exit_ids=())

    def test_class_or_query_validation_mismatch_fails(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
        sample = self._sample(points, 0, 1, 0, 1)
        with self.assertRaisesRegex(ValueError, "endpoint_class_differs"):
            self._certify(sample, points, {(0,4,7): "outside", (1,4,7): "outside"}, "inside")
        with self.assertRaisesRegex(ValueError, "complete 185-bank"):
            self._certify(sample, points, {(0,4,7): "outside", (1,4,7): "inside"}, "inside",
                          complete_sample_query_185=False)
        with self.assertRaisesRegex(ValueError, "query identities"):
            self._certify(sample, points, {(0,4,7): "outside", (1,4,7): "inside"}, "inside",
                          endpoint_query_identity_valid=False)


if __name__ == "__main__":
    unittest.main()
