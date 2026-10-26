from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path("/Users/n/Work/numi-human-connected-surface-069")
OWNER = ROOT / "barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py"
CANDIDATE = ROOT / "barycentric-connected-clearance-170/coherent-support-continuation-362/uniform-scale-recovery/backtrack/pass-001/candidate.npz"
PAIRS = ROOT / "barycentric-connected-clearance-170/coherent-support-continuation-382/round-001/attempt-001/self-pairs-input.npy"
OWNER_SHA = "c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3"
CANDIDATE_SHA = "814b879cb8f3af988aeaa0540d183222d942e95f5597434b6328f714d35e14d3"
PAIRS_SHA = "6ff1d81071cb36b81969fcb03ba94e539915ba2e4ed0c6457e99f6e9e790bbe2"

spec = importlib.util.spec_from_file_location("row_f32_precision_audit", HERE / "row_f32_precision_audit.py")
precision = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(precision)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_owner():
    spec = importlib.util.spec_from_file_location("pinned_self_separation_026_test", OWNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrecisionAuditTests(unittest.TestCase):
    def test_symmetric_half_ulp_is_conservative_at_binade_boundary(self):
        p = np.asarray([[1.0, -1.0, 0.0]], dtype=np.float32)
        h = precision._symmetric_half_ulp_mm(p)
        # At +/-1 the toward-zero spacing is half the away-from-zero spacing.
        self.assertAlmostEqual(h[0, 0], (2.0 ** -24) * 1000.0, places=12)
        self.assertAlmostEqual(h[0, 1], (2.0 ** -24) * 1000.0, places=12)
        self.assertGreater(h[0, 2], 0.0)

    def test_interval_bound_covers_larger_spacing_across_power_of_two(self):
        p = np.asarray([[1.0 - 2.0 ** -23, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        row = {"nodes": np.array([0, 1]), "coefficients": np.array([[1.,0.,0.],[-1.,0.,0.]]), "lower": -1.0, "label": "3:8:A0"}
        rec = [{"faces": [3,8], "kind": "moving_sat_midplane", "preserved_margin_mm": 1.0}]
        current = precision.audit_pair_rows_f32_precision(p, [row], rec)
        interval = precision.audit_pair_rows_f32_precision(p, [row], rec, interval_radius_mm=0.001)
        self.assertGreaterEqual(interval["rows"][0]["rounding_bound_within_axiswise_interval_mm"], current["rows"][0]["rounding_bound_at_current_f32_mm"])
        self.assertEqual(interval["summary"]["coordinate_scope"], "axiswise cube around emitted F32 coordinates")

    def test_reports_subprecision_and_kind_without_changing_rows(self):
        p = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        rows = [
            {"nodes": np.array([0, 1]), "coefficients": np.array([[1., 0., 0.], [-1., 0., 0.]]), "lower": 1e-4, "label": "3:8:A0"},
            {"nodes": np.array([0, 1]), "coefficients": np.array([[1., 0., 0.], [-1., 0., 0.]]), "lower": 1.0, "label": "5:9:A0"},
        ]
        before = [r["lower"] for r in rows]
        records = [
            {"faces": [3, 8], "kind": "moving_sat_midplane", "preserved_margin_mm": 1e-4},
            {"faces": [5, 9], "kind": "common_vertex_wedge", "preserved_margin_mm": 1.0},
        ]
        result = precision.audit_pair_rows_f32_precision(p, rows, records)
        self.assertTrue(result["rows"][0]["current_subprecision_margin"])
        self.assertFalse(result["rows"][1]["current_subprecision_margin"])
        self.assertEqual(result["rows"][0]["pair_kind"], "moving_sat_midplane")
        self.assertEqual(result["summary"]["current_subprecision_rows"], 1)
        self.assertEqual(before, [r["lower"] for r in rows])

    def test_actual_13_seeded_disjoint_pairs_around_face_11640(self):
        self.assertEqual(sha256(OWNER), OWNER_SHA)
        self.assertEqual(sha256(CANDIDATE), CANDIDATE_SHA)
        self.assertEqual(sha256(PAIRS), PAIRS_SHA)
        with np.load(CANDIDATE, allow_pickle=False) as z:
            points = z["points_world_m"]
            faces = z["source_O_faces"].astype(np.int64, copy=False)
        pairs = np.load(PAIRS, allow_pickle=False)
        pairs = np.sort(pairs[np.any(pairs == 11640, axis=1)], axis=1)
        self.assertEqual(pairs.shape, (13, 2))
        owner = load_owner()
        rows, records = owner.build_pair_rows(points.astype(np.float64) * 1000.0, faces, pairs, margin_fraction=0.25)
        self.assertEqual(len(records), 13)
        self.assertTrue(all(r["kind"] == "moving_sat_midplane" for r in records))
        result = precision.audit_pair_rows_f32_precision(points, rows, records)
        self.assertEqual(result["summary"]["pair_count_with_rows"], 13)
        self.assertEqual(result["summary"]["row_count"], 78)
        self.assertFalse(result["summary"]["unmatched_row_pairs"])
        self.assertFalse(result["summary"]["pair_records_without_rows"])
        self.assertGreater(result["summary"]["maximum_rounding_bound_mm"], 0.0)

    def test_rejects_unmerged_duplicate_nodes_and_non_f32_points(self):
        p = np.zeros((2, 3), dtype=np.float32)
        bad = {"nodes": np.array([0, 0]), "coefficients": np.array([[1., 0., 0.], [-1., 0., 0.]]), "lower": 1., "label": "0:1:A0"}
        pair_record = [{"faces": [0, 1], "kind": "moving_sat_midplane", "preserved_margin_mm": 1.0}]
        with self.assertRaisesRegex(ValueError, "not merged"):
            precision.audit_pair_rows_f32_precision(p, [bad], pair_record)
        with self.assertRaisesRegex(ValueError, "float32"):
            precision.audit_pair_rows_f32_precision(p.astype(np.float64), [])

    def test_finite_float32_limit_fails_closed(self):
        p = np.asarray([[np.finfo(np.float32).max, 0.0, 0.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "finite-float32 limit"):
            precision._symmetric_half_ulp_mm(p)

    def test_rejects_bad_metadata_and_bad_row_shape(self):
        p = np.zeros((2, 3), dtype=np.float32)
        row = {"nodes": np.array([0, 1]), "coefficients": np.ones((2, 2)), "lower": 1., "label": "0:1:A0"}
        with self.assertRaisesRegex(ValueError, r"finite \(len\(nodes\),3\)"):
            precision.audit_pair_rows_f32_precision(p, [row], [{"faces": [0, 1], "kind": "moving_sat_midplane", "preserved_margin_mm": 1.0}])
        good = {"nodes": np.array([0, 1]), "coefficients": np.array([[1.,0.,0.],[-1.,0.,0.]]), "lower": 1., "label": "0:1:A0"}
        with self.assertRaisesRegex(ValueError, "unknown 026 pair kind"):
            precision.audit_pair_rows_f32_precision(p, [good], [{"faces": [0,1], "kind": "other", "preserved_margin_mm": 1.0}])


if __name__ == "__main__":
    unittest.main()
