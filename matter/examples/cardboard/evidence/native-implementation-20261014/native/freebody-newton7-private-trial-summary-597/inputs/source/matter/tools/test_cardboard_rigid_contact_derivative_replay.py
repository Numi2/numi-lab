#!/usr/bin/env python3
"""Regression checks for the source-bound capsule-contact derivative replay.

These tests validate the captured local derivative diagnosis and FP32 update
replay. They do not execute Metal or qualify a complete mechanics run.
"""
from pathlib import Path
import unittest

from cardboard_rigid_contact_derivative_replay import (
    DEFAULT_BUILD,
    DEFAULT_CAPTURE,
    DEFAULT_PREDICTION,
    DEFAULT_SNAPSHOT,
    add,
    barrier,
    build_report,
    capsule_side_feature_admissible,
    capsule_side_geometric_action,
    capsule_side_operator_action,
    contact_residual,
    decode_armijo_result,
    dot,
    norm,
    rel_norm_error,
    scale,
    strict_inertial_stiffness_branch,
    sub,
)


ACCEPTED_CAPTURE = Path(
    "/Users/home/cardboard-implementation-20261014/native/"
    "capsule-side-bounded-capture-045/stderr.log"
)
ACCEPTED_SNAPSHOT = Path(
    "/Users/home/cardboard-implementation-20261014/snapshots/"
    "capsule-side-tangent-030"
)
ACCEPTED_BUILD = Path(
    "/Users/home/numi-cardboard-20261006/build-cardboard-provenance-030"
)


class CapsuleContactDerivativeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report(
            DEFAULT_CAPTURE,
            DEFAULT_PREDICTION,
            DEFAULT_SNAPSHOT,
            DEFAULT_BUILD,
            root=0,
            iteration=42,
        )
        cls.contacts = {row["node_id"]: row for row in cls.report["contacts"]}

    def test_source_bound_smooth_side_fd_matches_full_capsule_hessian(self):
        self.assertEqual(
            self.report["capture"]["same_candidate_solution_samples_across_stages"],
            True,
        )
        for node_id in (22, 29):
            row = self.contacts[node_id]
            self.assertFalse(row["captured_includes_capsule_geometry"])
            self.assertEqual(row["stiffness_branch"], "inertial")
            eps001 = next(
                item for item in row["central_difference"]
                if item["epsilon_velocity_scale"] == 0.001
            )
            self.assertLess(
                eps001["relative_error_vs_full_capsule_hessian"], 1.0e-6
            )
            self.assertGreater(
                eps001["relative_error_vs_expected_captured_action"], 1.0e-5
            )

    def test_capture_contains_authoritative_reference_displacement(self):
        for node_id in (22, 29):
            state = self.contacts[node_id]["candidate_state_arithmetic"]
            self.assertEqual(state["reference_displacement_mode"], 1.0)
            self.assertEqual(state["captured_minus_rest_plus_u_m"], [0.0] * 3)
            self.assertLessEqual(
                max(abs(x) for x in state["u_minus_dt_times_candidate_velocity_m"]),
                1.0e-12,
            )
            self.assertGreater(
                abs(state["capsule_gap_difference_rebased_minus_world_m"]),
                1.0e-11,
            )

    def test_final_armijo_alpha_is_unrepresentable_in_candidate_state(self):
        self.assertAlmostEqual(
            self.report["fp32_armijo_representability"]["recorded_last_trial_alpha"],
            4.24359969e-9,
            delta=1.0e-15,
        )
        for node in self.report["fp32_armijo_representability"]["nodes"]:
            last = node["trials"][-1]
            self.assertTrue(last["velocity_bits_unchanged"])
            self.assertTrue(last["position_bits_unchanged"])
            self.assertTrue(last["gap_bits_unchanged"])

    def test_node29_capsule_geometry_is_material_to_local_action(self):
        self.assertGreater(
            self.contacts[29]["geometric_action_over_radial_action"], 0.018
        )
        self.assertLess(
            self.contacts[22]["geometric_action_over_radial_action"], 0.001
        )


class AcceptedNativeCapsuleContactReplayTests(unittest.TestCase):
    """The accepted local trial is distinct from the failed full-run receipt."""

    @classmethod
    def setUpClass(cls):
        cls.report = build_report(
            ACCEPTED_CAPTURE,
            ACCEPTED_CAPTURE.with_name("prediction.json"),
            ACCEPTED_SNAPSHOT,
            ACCEPTED_BUILD,
            root=0,
            iteration=42,
        )

    def test_accepted_alpha_schedule_and_run_failure_are_separate(self):
        capture = self.report["capture"]
        result = capture["armijo_interpretation"]
        self.assertEqual(result["outcome"], "accepted_trial")
        self.assertTrue(result["accepted"])
        self.assertFalse(result["certified_noop"])
        self.assertEqual(result["trial_count"], 16)
        self.assertAlmostEqual(result["selected_alpha"], 1.05650986e-6, delta=1e-12)
        self.assertAlmostEqual(result["alpha_schedule"][-1], result["selected_alpha"], delta=1e-15)

        env = capture["armijo_record"]["environments"][0]
        self.assertEqual(env["solver_status_code"], 0)
        self.assertEqual(env["trial_status"]["code"], 10)
        self.assertFalse(result["trial_status_code_is_authoritative"])
        # Native process exit 1 records that the later whole run did not
        # complete successfully; it does not negate this accepted Newton step.
        self.assertEqual(self.report["native_run"]["exit_code"], 1)
        self.assertFalse(self.report["native_run"]["timed_out"])

    def test_captured_two_contact_action_matches_independent_full_hessian(self):
        self.assertTrue(all(row["captured_includes_capsule_geometry"]
                            for row in self.report["contacts"]))
        for row in self.report["contacts"]:
            self.assertLess(row["captured_vs_full_capsule_action_relative_error"], 1.0e-6)
            eps = next(item for item in row["central_difference"]
                       if item["epsilon_velocity_scale"] == 0.001)
            self.assertEqual(eps["plus_stiffness_branch"], "inertial")
            self.assertEqual(eps["minus_stiffness_branch"], "inertial")
            self.assertLess(eps["relative_error_vs_full_capsule_hessian"], 1.0e-6)
        total = self.report["selected_contacts_total_action"]
        self.assertEqual(total["contact_node_ids"], [22, 29])
        self.assertLess(total["relative_error"], 1.0e-6)

    def test_selected_alpha_is_included_in_binary32_representability_sweep(self):
        fp32 = self.report["fp32_armijo_representability"]
        self.assertEqual(fp32["outcome"], "accepted_trial")
        self.assertEqual(fp32["recorded_trial_count"], 16)
        for node in fp32["nodes"]:
            self.assertEqual(node["reference_displacement_mode"], 1.0)
            self.assertEqual(node["trials"][-1]["alpha"], fp32["selected_alpha"])


class PortableCapsuleSideDerivativeTests(unittest.TestCase):
    """Synthetic CPU check independent of the sealed forensic capture."""

    def test_smooth_side_fd_matches_signed_full_action_not_radial_only(self):
        axis = [0.0, 1.0, 0.0]
        normal = [0.6, 0.0, 0.8]
        direction = [0.4, 0.2, -0.7]
        gap = 2.0e-4
        radius = 5.0e-4
        dt = 1.0e-4
        thickness = 1.0e-2
        inverse_response = 1.0
        current_velocity = scale(-0.1, normal)
        tool_velocity = [0.0, 0.0, 0.0]

        self.assertTrue(capsule_side_feature_admissible(0.5, normal, axis))
        self.assertTrue(strict_inertial_stiffness_branch(
            gap, thickness, inverse_response, dt, 0.1))
        base = barrier(gap, thickness, inverse_response, dt, 0.1)
        radial_only = scale(
            base["radial_velocity_hessian"] * dot(normal, direction), normal)
        full = capsule_side_operator_action(
            normal, axis, base["radial_velocity_hessian"], base["impulse"],
            gap + radius, dt, direction)
        signed_geometry = capsule_side_geometric_action(
            normal, axis, base["impulse"], gap + radius, dt, direction)
        self.assertGreater(norm(signed_geometry), 0.1 * norm(radial_only))
        self.assertGreater(norm(sub(full, radial_only)), 0.1 * norm(radial_only))

        for epsilon in (1.0e-3, 1.0e-4):
            plus = contact_residual(
                direction, epsilon, current_velocity, tool_velocity, gap,
                normal, radius, dt, thickness, inverse_response, axis)
            minus = contact_residual(
                direction, -epsilon, current_velocity, tool_velocity, gap,
                normal, radius, dt, thickness, inverse_response, axis)
            self.assertTrue(strict_inertial_stiffness_branch(
                plus[1], thickness, inverse_response, dt,
                max(-plus[3], 0.0)))
            self.assertTrue(strict_inertial_stiffness_branch(
                minus[1], thickness, inverse_response, dt,
                max(-minus[3], 0.0)))
            fd_action = scale(-0.5 / epsilon, sub(plus[0], minus[0]))
            self.assertLess(rel_norm_error(fd_action, full), 1.0e-6)
            self.assertGreater(rel_norm_error(fd_action, radial_only), 0.1)

    def test_capsule_endpoint_and_stiffness_kink_are_excluded(self):
        axis = [0.0, 1.0, 0.0]
        side_normal = [1.0, 0.0, 0.0]
        self.assertFalse(capsule_side_feature_admissible(1.0e-3, side_normal, axis))
        self.assertFalse(capsule_side_feature_admissible(1.0 - 1.0e-3, side_normal, axis))
        self.assertFalse(capsule_side_feature_admissible(0.5, [0.0, 1.0, 0.0], axis))
        self.assertFalse(capsule_side_feature_admissible(float("nan"), side_normal, axis))

        # For these values both stiffness formulae equal 10 exactly. The
        # branch must remain excluded at the max() kink and within its guard.
        tie = (0.5, 1.0, 1.0, 1.0, 5.0)
        self.assertFalse(strict_inertial_stiffness_branch(*tie))
        self.assertFalse(strict_inertial_stiffness_branch(
            tie[0], tie[1], tie[2], tie[3], 5.0 - 1.0e-5))
        self.assertTrue(strict_inertial_stiffness_branch(
            tie[0], tie[1], tie[2], tie[3], 5.0 - 1.0e-3))

    def test_armijo_decoder_handles_accepted_and_failed_records(self):
        accepted = decode_armijo_result({
            "trial_alpha": 0.125,
            "selected_alpha": 0.125,
            "accepted": True,
            "certified_noop": False,
            "line_search_raw": [0.125, 0.99, 4.0, 42.0],
            "solver_status": {"diagnostics": [None, 0.0, 0.0, 0.0]},
            # The capture may contain this later sentinel after selection.
            "trial_status": {"code": 10},
        })
        self.assertEqual(accepted["outcome"], "accepted_trial")
        self.assertEqual(accepted["alpha_schedule"], [1.0, 0.5, 0.25, 0.125])
        self.assertFalse(accepted["trial_status_code_is_authoritative"])

        failed = decode_armijo_result({
            "trial_alpha": 0.125,
            "selected_alpha": 0.0,
            "accepted": False,
            "certified_noop": False,
            "line_search_raw": [0.0, 0.99, 1.0, 42.0],
            "solver_status": {"diagnostics": [1.0, 1.0, 0.125, 4.0]},
        })
        self.assertEqual(failed["outcome"], "rejected_line_search")
        self.assertEqual(failed["alpha_schedule"], [1.0, 0.5, 0.25, 0.125])


if __name__ == "__main__":
    unittest.main()
