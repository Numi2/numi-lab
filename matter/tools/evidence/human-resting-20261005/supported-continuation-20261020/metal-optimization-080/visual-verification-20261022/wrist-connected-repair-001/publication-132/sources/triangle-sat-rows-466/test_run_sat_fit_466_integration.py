from pathlib import Path
import importlib.util
import sys
import types
import unittest

import numpy as np

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
FIT = D / "D/D/triangle-sat-rows-466/run_sat_fit_466.py"
SAT = D / "D/D/triangle-sat-rows-466/triangle_sat_rows_466.py"
WRAPPER = D / "D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py"
OWNER = D / "D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SatFit466AdapterIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load("sat_fit_466_adapter_test", FIT)
        cls.sat = load("triangle_sat_rows_466_for_adapter_test", SAT)
        cls.wrapper = load("pinned_fit458_for_sat466_adapter_test", WRAPPER)
        cls.owner = load("fitter408_for_sat466_adapter_test", OWNER)
        cls.adapter.sat = cls.sat
        cls.adapter.original_install = cls.wrapper.install_objective

    def test_pinned_458_install_path_preserves_point_prefix_and_injects_sat_guards(self):
        points = np.array([
            [-1.0, 0.05, 1.0],
            [1.0, 0.20, 1.0],
            [0.0, 0.05, 2.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        support = np.array([0, 1, 2], dtype=np.int64)
        anchors = np.array([], dtype=np.int64)
        bone = np.array([[0.0, 0.0, 0.0],
                         [0.0, 0.1, 0.0],
                         [0.0, 0.0, 10.0]], dtype=np.float64)
        pairs = [
            dict(face_row=0, body=43, bone_id=16, triangle_id=133),
            dict(face_row=0, body=44, bone_id=17, triangle_id=134),
        ]
        bones = {(43, 16, 133): bone, (44, 17, 134): bone.copy()}
        pair_identity = np.array([[43, 16, 0, 133],
                                  [44, 17, 0, 134]], dtype=np.int64)
        crossing = {
            "pairs": pairs,
            "bone_triangles_mm": bones,
            "pair_identity_array": pair_identity,
        }

        point_targets = {
            "point_rows": np.array([0], dtype=np.int64),
            "point_targets_f32_m": np.array([points[0] / 1000.0 + [0.001, 0.0, 0.0]],
                                            dtype=np.float32),
            "point_directions": np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        }
        barycentric = np.array([[1.0 / 3.0] * 3], dtype=np.float64)
        sample_mm = (barycentric[0, :, None] * points[faces[0]]).sum(axis=0)
        face_targets = {
            "face_rows": np.array([0, 0], dtype=np.int64),
            "barycentric_weights": np.repeat(barycentric, 2, axis=0),
            "face_targets_f32_m": np.repeat(
                np.array([(sample_mm + [1.0, 0.0, 0.0]) / 1000.0], dtype=np.float32), 2, axis=0),
            "face_directions": np.repeat(np.array([[1.0, 0.0, 0.0]]), 2, axis=0),
            "face_samples_m": np.repeat(np.array([sample_mm / 1000.0]), 2, axis=0),
            "face_npz": {
                "bank_body": np.array([43, 44], dtype=np.int64),
                "bank_stable_bone_id": np.array([16, 17], dtype=np.int64),
                "bank_triangle_stable_id": np.array([133, 134], dtype=np.int64),
            },
        }

        def build_barycentric_exit_system(p, f, face_rows, weights, directions,
                                          targets, anchor_rows=()):
            tri = np.asarray(f, dtype=np.int64)[face_rows]
            B = np.zeros((len(face_rows), len(p)), dtype=np.float64)
            for i, nodes in enumerate(tri):
                B[i, nodes] = weights[i]
            coeff = weights[:, :, None] * directions[:, None, :]
            samples = np.asarray(B @ p)
            h = np.einsum("ij,ij->i", directions, targets * 1000.0 - samples)
            return B, tri, coeff, h, None

        bary_owner = types.SimpleNamespace(
            build_barycentric_exit_system=build_barycentric_exit_system)

        # Record the genuine 408 output before 458 installs its replacement.
        base_C, base_h, base_w, base_meta = self.owner.build_clearance_rows(
            points, faces, support, anchors, point_targets, face_targets, bary_owner)

        captured = []
        original_sat_builder = self.sat.build_fixed_bone_outward_plane_rows
        def capture_sat_builder(*args, **kwargs):
            result = original_sat_builder(*args, **kwargs)
            captured.append(result)
            return result
        self.sat.build_fixed_bone_outward_plane_rows = capture_sat_builder
        state = self.adapter.install_sat(
            self.owner, self.wrapper.install_objective, crossing)
        result_C, result_h, result_w, result_meta = self.owner.build_clearance_rows(
            points, faces, support, anchors, point_targets, face_targets, bary_owner)
        self.sat.build_fixed_bone_outward_plane_rows = original_sat_builder

        # The 408 point target remains the exact byte-equivalent first row.
        np.testing.assert_array_equal(result_C[:1].toarray(), base_C[:1].toarray())
        np.testing.assert_array_equal(result_h[:1], base_h[:1])
        np.testing.assert_array_equal(result_w[:1], base_w[:1])
        self.assertEqual(result_meta[:1], base_meta[:1])
        self.assertEqual(state["objective"]["point_row_count"], 1)
        self.assertEqual(state["objective"]["point_weight_mass"], 0.5)

        # Both current typed crossing pairs use SAT, with three hard lower
        # bounds per pair and the chosen axis metadata carried to each row.
        self.assertEqual(state["objective"]["all_current_crossing_pairs_loaded"], 2)
        self.assertEqual(state["objective"]["three_corner_lower_bounds_per_pair"], 6)
        self.assertEqual(state["objective"]["positive_bone_plane_target_rows"], 4)
        self.assertEqual(result_C.shape[0], 1 + 4)
        self.assertEqual(len(result_h), 5)
        self.assertEqual(len(result_w), 5)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["diagnostics"]["triangle_sat_axis_pair_count"], 2)
        self.assertEqual(len(captured[0]["guard_metadata"]), 6)
        all_row_metadata = captured[0]["guard_metadata"] + result_meta[1:]
        self.assertTrue(all(
            row["axis_strategy"] == "triangle_sat_minimum_max_deficit"
            and row["axis_source"] == "edge_cross"
            and row["candidate_axis_count"] == 13
            and row["axis_outward_dot"] >= 0.0
            for row in all_row_metadata))
        # One negative corner from each pair becomes a nonpositive hard guard.
        self.assertEqual(len(state["hard_rows"]), 2)
        self.assertTrue(all(row["kind"] ==
                            "fixed_bone_outward_plane_nonregression_458"
                            and row["lower"] <= 0.0
                            for row in state["hard_rows"]))
        self.assertEqual(
            [row["typed_ids"][:4].tolist() for row in state["hard_rows"]],
            [[0, 43, 16, 133], [0, 44, 17, 134]])

        combined = self.owner.combine_barrier_rows(
            [{"kind": "self"}], fixed_boundary_rows=[{"kind": "fixed"}])
        self.assertEqual([row["kind"] for row in combined[:2]], ["self", "fixed"])
        self.assertEqual(combined[2:], state["hard_rows"])


if __name__ == "__main__":
    unittest.main()
