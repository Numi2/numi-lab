#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from metalrobo.policy_selection import (  # noqa: E402
    compare_evidence,
    evaluation_arguments,
)


class PolicySelectionTest(unittest.TestCase):
    def test_catastrophic_get_up_candidate_never_advances(self) -> None:
        incumbent = {
            "task": "supine-get-up",
            "termination_count": 170,
            "termination_count_by_environment": [0] * 86 + [1] * 170,
            "failed_environment_steps": 0,
            "mean_tilt": 0.137,
            "squat_cycle_completed_environment_rate": 0.0,
            "squat_cycle_evidence_by_environment": [],
        }
        candidate = {
            **incumbent,
            "termination_count": 256,
            "termination_count_by_environment": [1] * 256,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")
        self.assertIn("termination rate increased", decision["regressions"])
        self.assertTrue(decision["candidate_retained"])

    def test_partial_physical_progress_can_advance(self) -> None:
        incumbent = {
            "task": "supine-get-up",
            "termination_count": 80,
            "termination_count_by_environment": [0] * 20 + [1] * 80,
            "failed_environment_steps": 0,
            "mean_tilt": 0.2,
            "squat_cycle_completed_environment_rate": 0.1,
        }
        candidate = {
            **incumbent,
            "termination_count": 60,
            "termination_count_by_environment": [0] * 40 + [1] * 60,
            "squat_cycle_completed_environment_rate": 0.15,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "candidate")
        self.assertTrue(decision["candidate_advanced_deployment"])

    def test_horizon_timeouts_are_not_physical_terminations(self) -> None:
        incumbent = {
            "task": "velocity",
            "termination_count": 154,
            "timeout_count": 154,
            "height_or_tilt_termination_count": 0,
            "termination_count_by_environment": [0] * 102 + [1] * 154,
            "failed_environment_steps": 0,
            "mean_tracking_score": 0.52,
            "mean_tilt": 0.35,
            "mean_root_height": 0.60,
        }
        candidate = {
            **incumbent,
            "termination_count": 200,
            "timeout_count": 200,
            "mean_tracking_score": 0.53,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "candidate")
        self.assertEqual(
            decision["metrics"]["candidate_termination_rate"], 0.0
        )

    def test_tracking_gain_cannot_buy_collapsed_root_height(self) -> None:
        incumbent = {
            "task": "velocity",
            "termination_count": 0,
            "termination_count_by_environment": [0] * 256,
            "failed_environment_steps": 0,
            "mean_tracking_score": 0.52,
            "mean_tilt": 0.35,
            "mean_root_height": 0.60,
        }
        candidate = {
            **incumbent,
            "mean_tracking_score": 0.54,
            "mean_root_height": 0.55,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")
        self.assertIn("mean root height decreased", decision["regressions"])

    def test_forward_reach_is_explicit_progress(self) -> None:
        incumbent = {
            "task": "velocity",
            "termination_count": 0,
            "termination_count_by_environment": [0] * 256,
            "failed_environment_steps": 0,
            "forward_progress_available": True,
            "mean_peak_forward_progress_m": 0.42,
            "mean_tracking_score": 0.52,
            "mean_tilt": 0.35,
            "mean_root_height": 0.60,
        }
        candidate = {
            **incumbent,
            "mean_peak_forward_progress_m": 0.45,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "candidate")
        self.assertIn(
            "mean peak forward progress increased",
            decision["improvements"],
        )

    def test_one_deterministic_completion_is_progress(self) -> None:
        incumbent = {
            "task": "supine-get-up",
            "termination_count": 162,
            "termination_count_by_environment": [0] * 94 + [1] * 162,
            "failed_environment_steps": 0,
            "mean_tilt": 0.116925,
            "squat_cycle_completed_environment_rate": 14 / 256,
        }
        candidate = {
            **incumbent,
            "termination_count": 160,
            "termination_count_by_environment": [0] * 96 + [1] * 160,
            "mean_tilt": 0.116919,
            "squat_cycle_completed_environment_rate": 15 / 256,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "candidate")
        self.assertEqual(decision["regressions"], [])

    def test_zero_authority_teacher_is_removed_for_student_evaluation(self) -> None:
        arguments = evaluation_arguments(
            [
                "--task",
                "supine-get-up",
                "--envs",
                "1024",
                "--steps",
                "256",
                "--interaction-pack",
                "teacher.interactionpack",
                "--interaction-clip",
                "stand",
                "--interaction-student-authority",
                "0",
                "--interaction-reset-phase-fraction",
                "0.8",
            ],
            policy_pack=Path("candidate.policypack"),
            metallib=Path("MetalRobo.metallib"),
            state_trace=Path("candidate.tsv"),
            maximum_environments=256,
            held_out_seed=42,
            evaluation_steps=103,
        )
        self.assertIn("--interaction-reset-only", arguments)
        self.assertNotIn("--interaction-student-authority", arguments)
        self.assertNotIn("--interaction-reset-phase-fraction", arguments)
        environment_index = len(arguments) - 1 - arguments[::-1].index(
            "--envs"
        )
        seed_index = len(arguments) - 1 - arguments[::-1].index("--seed")
        self.assertEqual(arguments[environment_index + 1], "256")
        self.assertEqual(arguments[seed_index + 1], "42")
        steps_index = len(arguments) - 1 - arguments[::-1].index("--steps")
        self.assertEqual(arguments[steps_index + 1], "103")


if __name__ == "__main__":
    unittest.main()
