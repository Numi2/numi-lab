import unittest
import numpy as np

from build_quality_feasible_failed_pose554 import (
    combine_components,
    emit_component,
    payload_with_points,
    recorded_nongeometry_pass,
    select_component_alpha,
)


class FailedPose554Tests(unittest.TestCase):
    def test_self_failure_does_not_disqualify_recorded_quality_alpha(self):
        base = np.zeros((3, 3), dtype=np.float32)
        delta = np.ones((3, 3), dtype=np.float64)
        trials = [
            {"alpha": 1.0, "area_pass": True, "maximum_step_mm": .8,
             "maximum_source_distance_mm": 11.0, "minimum_edge_slack_mm": 0.0,
             "pass": False, "exact_self_pairs": 17},
            {"alpha": .5, "area_pass": True, "maximum_step_mm": .4,
             "maximum_source_distance_mm": 11.0, "minimum_edge_slack_mm": 0.0,
             "pass": True, "exact_self_pairs": 0},
        ]
        chosen, proof = select_component_alpha(
            base, delta, np.array([0, 1, 2]), {"trials": trials}, 1.0,
            lambda _p: {"pass": True},
        )
        self.assertEqual(chosen, 1.0)
        self.assertFalse(proof["self_intersection_audit_performed"])

    def test_recorded_area_failure_is_rejected_even_if_legacy_pass_is_true(self):
        ok, reasons = recorded_nongeometry_pass({
            "alpha": .5, "area_pass": False, "maximum_step_mm": .2,
            "maximum_source_distance_mm": 2.0, "minimum_edge_slack_mm": .1,
            "pass": True, "exact_self_pairs": 0,
        }, 1.0)
        self.assertFalse(ok)
        self.assertIn("recorded_area", reasons)

    def test_emission_uses_owner_order_and_changes_only_component(self):
        base = np.array([[.001, .002, .003], [.004, .005, .006]], dtype=np.float32)
        delta = np.array([[.123456789, -.2, .3], [.7, .8, -.9]], dtype=np.float64)
        out = emit_component(base, delta, np.array([1]), .25)
        expected = base.copy()
        base_mm = base.astype(np.float64) * 1000.0
        expected[1] = ((base_mm[1] + .25 * delta[1]) / 1000.0).astype(np.float32)
        self.assertEqual(out[0].tobytes(), base[0].tobytes())
        self.assertEqual(out.tobytes(), expected.tobytes())

    def test_component_combination_and_nonpoint_payload_preservation(self):
        base = np.zeros((4, 3), dtype=np.float32)
        delta = np.ones((4, 3), dtype=np.float64)
        groups = [np.array([0, 1]), np.array([2])]
        out = combine_components(base, delta, groups, [.5, .25])
        self.assertTrue(np.all(out[[0, 1]] == np.float32(.0005)))
        self.assertTrue(np.all(out[2] == np.float32(.00025)))
        self.assertTrue(np.all(out[3] == 0))
        payload = {"points_world_m": base, "source_O_faces": np.array([[0, 1, 2]], dtype=np.int32),
                   "typed_rows": np.array([3], dtype=np.int64)}
        new_payload = payload_with_points(payload, out)
        self.assertIs(new_payload["source_O_faces"], payload["source_O_faces"])
        self.assertEqual(new_payload["typed_rows"].tobytes(), payload["typed_rows"].tobytes())


if __name__ == "__main__":
    unittest.main()

