import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = ROOT / "D"
VARIANT = D / "D/fixed-bone-outward-plane-458"
BASE = D / "constraint-project-backtrack-456/run-001/backtrack-373/pass-006/candidate.npz"
PREP = D / "connected-continuation-prep-457/run-001"
FACE_NPZ = PREP / "surface-targets-335/targets.npz"
FACE_REPORT = PREP / "surface-targets-335/report.json"
POINT_NPZ = PREP / "point-targets-390/nearest-union-exits.npz"
POINT_REPORT = PREP / "point-targets-390/nearest-union-exits.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Matched408Integration458(unittest.TestCase):
    def test_current_matched_inputs_reach_408_builder_and_preserve_point_block(self):
        runner = load("runner458_integration_test", VARIANT / "run_grouped_fit_458.py")
        owner = load("owner408_integration_test", runner.OWNER)
        objective = load("objective458_integration_test", runner.HELPER)
        crossing = runner.prepare_crossing_inputs(BASE, BASE.parent / "audit-159/report.json")
        with np.load(BASE, allow_pickle=False) as z:
            points_f32 = np.asarray(z["points_world_m"], dtype=np.float32)
            anchors = np.asarray(z["typed_coarse_anchor_output_rows_018"], dtype=np.int64)
        point_targets, face_targets = owner.load_clearance_target_inputs(
            POINT_NPZ, POINT_REPORT, FACE_NPZ, FACE_REPORT,
            BASE, points_f32, crossing["faces"])
        bary = owner.load_module("barycentric_owner_458_integration_test", owner.DEFAULT_BARY_OWNER)
        support = np.unique(np.concatenate((
            point_targets["point_rows"], face_targets["face_nodes"].ravel()
        ))).astype(np.int64)
        points_mm = points_f32.astype(np.float64) * 1000.0
        original = owner.build_clearance_rows
        expected_C, expected_h, expected_w, expected_meta = original(
            points_mm, crossing["faces"], support, anchors,
            point_targets, face_targets, bary)
        point_count = len(point_targets["point_rows"])
        state = runner.install_objective(owner, objective, crossing)
        C, h, w, meta = owner.build_clearance_rows(
            points_mm, crossing["faces"], support, anchors,
            point_targets, face_targets, bary)
        np.testing.assert_array_equal(C[:point_count].indptr, expected_C[:point_count].indptr)
        np.testing.assert_array_equal(C[:point_count].indices, expected_C[:point_count].indices)
        np.testing.assert_array_equal(C[:point_count].data, expected_C[:point_count].data)
        np.testing.assert_array_equal(h[:point_count], expected_h[:point_count])
        np.testing.assert_array_equal(w[:point_count], expected_w[:point_count])
        self.assertEqual(meta[:point_count], expected_meta[:point_count])
        self.assertEqual(state["objective"]["all_current_crossing_pairs_loaded"], 3136)
        self.assertEqual(state["objective"]["three_corner_lower_bounds_per_pair"], 9408)
        self.assertEqual(state["objective"]["anchored_positive_lower_count"], 0)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=10)
        hard = owner.combine_barrier_rows([], [])
        self.assertEqual(len(hard), state["objective"]["nonpositive_corner_hard_rows"])
        self.assertTrue(all(float(row["lower"]) <= 0.0 for row in hard))
        self.assertEqual(state["objective"]["nonpositive_corner_hard_rows"], 4518)
        self.assertEqual(sum(item["kind"] == "face" for item in meta[point_count:]), len(h) - point_count)


if __name__ == "__main__":
    unittest.main()

