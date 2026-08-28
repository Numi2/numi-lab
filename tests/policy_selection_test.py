#!/usr/bin/env python3

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import metalrobo.policy_selection as policy_selection  # noqa: E402
from metalrobo.policy_selection import (  # noqa: E402
    _adult_evaluation_bands,
    _curriculum_evaluation_bands,
    _crow_journey_contract_regressions,
    _deduplicate_policy_paths,
    _evaluate,
    compare_protected_bands,
    compare_staged_bands,
    compare_adult_bands,
    compare_evidence,
    evaluation_arguments,
    select_candidate_champion,
)


class PolicySelectionTest(unittest.TestCase):
    def test_exact_duplicate_policy_packs_are_evaluated_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "revision-00000050.candidate.policypack"
            final = root / "candidate.policypack"
            distinct = root / "revision-00000100.candidate.policypack"
            checkpoint.write_bytes(b"same policy")
            final.write_bytes(b"same policy")
            distinct.write_bytes(b"distinct policy")
            policies, duplicates = _deduplicate_policy_paths(
                [checkpoint, distinct, final]
            )
            self.assertEqual(policies, [checkpoint, distinct])
            self.assertEqual(duplicates, {str(final): str(checkpoint)})

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

    def test_imported_velocity_requires_a_clean_horizon_to_deploy(self) -> None:
        incumbent = {
            "task": "robotis_ai_sapiens_k1_velocity_v1",
            "world_source": "urdf",
            "termination_count": 800,
            "termination_count_by_environment": [50] * 16,
            "clean_horizon_environment_rate": 0.0,
            "failed_environment_steps": 0,
            "mean_reward": -0.09,
            "mean_task_reward": 0.0,
            "mean_tracking_score": 0.28,
        }
        candidate = {
            **incumbent,
            "termination_count": 736,
            "termination_count_by_environment": [46] * 16,
            "mean_reward": -0.08,
            "mean_tracking_score": 0.17,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")
        self.assertFalse(decision["candidate_advanced_deployment"])
        self.assertIn(
            "candidate completed no clean horizon", decision["regressions"]
        )

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

    def test_forward_progress_is_not_vetoed_by_balance_tradeoff(self) -> None:
        incumbent = {
            "task": "velocity",
            "termination_count_by_environment": [0] * 256,
            "height_or_tilt_termination_count": 0,
            "failed_environment_steps": 0,
            "forward_progress_available": True,
            "mean_peak_forward_progress_m": 1.88,
            "mean_final_forward_progress_m": 1.68,
            "mean_tracking_score": 0.526,
            "mean_root_height": 0.59,
            "mean_tilt": 0.38,
        }
        candidate = {
            **incumbent,
            "mean_peak_forward_progress_m": 2.04,
            "mean_final_forward_progress_m": 1.85,
            "mean_tracking_score": 0.534,
            "mean_root_height": 0.52,
            "mean_tilt": 0.56,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "candidate")
        self.assertGreater(decision["selection_score"], 0.0)
        self.assertIn("mean tilt increased", decision["regressions"])
        self.assertIn("mean root height decreased", decision["regressions"])

    def test_adult_promotion_cannot_buy_survivability_with_stillness(self) -> None:
        incumbent = {
            "task": "unitree_g1_adult_locomotion",
            "termination_count_by_environment": [0] * 256,
            "height_or_tilt_termination_count": 256,
            "failed_environment_steps": 0,
            "forward_progress_available": True,
            "mean_peak_forward_progress_m": 0.49,
            "mean_final_forward_progress_m": 0.14,
            "mean_tracking_score": 0.62,
            "mean_root_height": 0.705,
            "mean_tilt": 0.242,
            "outcomes": {
                "contact_reward": {"mean": -0.00039, "direction": 1},
                "standing_completion": {"mean": 0.309, "direction": 1},
                "restoration": {"mean": 0.164, "direction": 1},
            },
        }
        candidate = {
            **incumbent,
            "height_or_tilt_termination_count": 64,
            "mean_peak_forward_progress_m": 0.0,
            "mean_final_forward_progress_m": -0.09,
            "mean_tracking_score": 0.60,
            "mean_root_height": 0.718,
            "mean_tilt": 0.189,
            "outcomes": {
                "contact_reward": {"mean": -0.00005, "direction": 1},
                "standing_completion": {"mean": 0.335, "direction": 1},
                "restoration": {"mean": 0.168, "direction": 1},
            },
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["task"], "adult-locomotion")
        self.assertEqual(decision["selected"], "incumbent")
        self.assertEqual(
            decision["selection_method"],
            "adult_locomotion_physical_comparison",
        )
        self.assertIn(
            "mean peak forward progress decreased",
            decision["regressions"],
        )

    def test_adult_promotion_guards_authored_standing_outcomes(self) -> None:
        incumbent = {
            "task": "unitree_g1_adult_locomotion",
            "termination_count_by_environment": [0] * 256,
            "height_or_tilt_termination_count": 256,
            "failed_environment_steps": 0,
            "forward_progress_available": True,
            "mean_peak_forward_progress_m": 0.40,
            "mean_final_forward_progress_m": 0.30,
            "mean_tracking_score": 0.50,
            "mean_root_height": 0.70,
            "mean_tilt": 0.25,
            "outcomes": {
                "contact_reward": {"mean": -0.00040, "direction": 1},
                "standing_completion": {"mean": 0.30, "direction": 1},
                "restoration": {"mean": 0.16, "direction": 1},
            },
        }
        candidate = {
            **incumbent,
            "height_or_tilt_termination_count": 240,
            "mean_peak_forward_progress_m": 0.45,
            "mean_final_forward_progress_m": 0.34,
            "mean_tracking_score": 0.51,
            "mean_root_height": 0.71,
            "mean_tilt": 0.24,
            "outcomes": {
                "contact_reward": {"mean": -0.00035, "direction": 1},
                "standing_completion": {"mean": 0.28, "direction": 1},
                "restoration": {"mean": 0.17, "direction": 1},
            },
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")
        self.assertIn(
            "adult authored outcome standing_completion decreased",
            decision["regressions"],
        )

    def test_adult_locomotion_progress_and_survivability_can_advance(self) -> None:
        incumbent = {
            "task": "unitree_g1_adult_locomotion",
            "termination_count_by_environment": [0] * 256,
            "height_or_tilt_termination_count": 256,
            "failed_environment_steps": 0,
            "forward_progress_available": True,
            "mean_peak_forward_progress_m": 0.40,
            "mean_final_forward_progress_m": 0.30,
            "mean_tracking_score": 0.50,
            "mean_root_height": 0.70,
            "mean_tilt": 0.25,
            "outcomes": {
                "contact_reward": {"mean": -0.00040, "direction": 1},
                "standing_completion": {"mean": 0.30, "direction": 1},
                "restoration": {"mean": 0.16, "direction": 1},
            },
        }
        candidate = {
            **incumbent,
            "height_or_tilt_termination_count": 240,
            "mean_peak_forward_progress_m": 0.45,
            "mean_final_forward_progress_m": 0.34,
            "mean_tracking_score": 0.51,
            "mean_root_height": 0.71,
            "mean_tilt": 0.24,
            "outcomes": {
                "contact_reward": {"mean": -0.00035, "direction": 1},
                "standing_completion": {"mean": 0.31, "direction": 1},
                "restoration": {"mean": 0.17, "direction": 1},
            },
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "candidate")
        self.assertEqual(
            decision["selection_method"],
            "adult_locomotion_physical_comparison",
        )

    def test_adult_previous_band_regression_blocks_new_band_progress(self) -> None:
        incumbent = {
            "task": "unitree_g1_adult_locomotion",
            "termination_count_by_environment": [0] * 256,
            "height_or_tilt_termination_count": 256,
            "failed_environment_steps": 0,
            "forward_progress_available": True,
            "mean_peak_forward_progress_m": 0.40,
            "mean_final_forward_progress_m": 0.30,
            "mean_tracking_score": 0.50,
            "mean_root_height": 0.70,
            "mean_tilt": 0.25,
            "outcomes": {
                "contact_reward": {"mean": -0.00040, "direction": 1},
                "standing_completion": {"mean": 0.30, "direction": 1},
                "restoration": {"mean": 0.16, "direction": 1},
            },
        }
        current_candidate = {
            **incumbent,
            "height_or_tilt_termination_count": 240,
            "mean_peak_forward_progress_m": 0.45,
            "mean_final_forward_progress_m": 0.34,
            "mean_tracking_score": 0.51,
            "mean_root_height": 0.71,
            "mean_tilt": 0.24,
            "outcomes": {
                "contact_reward": {"mean": -0.00035, "direction": 1},
                "standing_completion": {"mean": 0.31, "direction": 1},
                "restoration": {"mean": 0.17, "direction": 1},
            },
        }
        previous_candidate = {
            **incumbent,
            "mean_root_height": 0.68,
            "mean_tilt": 0.30,
        }
        decision = compare_adult_bands(
            incumbent,
            current_candidate,
            incumbent,
            previous_candidate,
        )
        self.assertEqual(decision["selected"], "incumbent")
        self.assertIn(
            "previous-band: mean root height decreased",
            decision["regressions"],
        )

    def test_all_locomotion_checkpoints_compare_directly_to_incumbent(
        self,
    ) -> None:
        incumbent = {
            "task": "velocity",
            "termination_count_by_environment": [0] * 256,
            "height_or_tilt_termination_count": 0,
            "failed_environment_steps": 0,
            "forward_progress_available": True,
            "mean_peak_forward_progress_m": 1.95,
            "mean_final_forward_progress_m": 1.80,
            "mean_tracking_score": 0.525,
            "mean_root_height": 0.605,
            "mean_tilt": 0.352,
        }
        candidates = {
            "revision-206": {
                **incumbent,
                "mean_peak_forward_progress_m": 2.044,
                "mean_final_forward_progress_m": 1.852,
                "mean_tracking_score": 0.534,
                "mean_root_height": 0.521,
                "mean_tilt": 0.565,
            },
            "revision-162": {
                **incumbent,
                "mean_peak_forward_progress_m": 1.936,
                "mean_final_forward_progress_m": 1.780,
                "mean_root_height": 0.595,
                "mean_tilt": 0.371,
            },
        }
        champion, comparisons = select_candidate_champion(
            incumbent, candidates
        )
        self.assertEqual(champion, "revision-206")
        self.assertEqual(set(comparisons), set(candidates))

    def test_world_pack_selection_uses_authored_task_outcome(self) -> None:
        incumbent = {
            "task": "velocity",
            "world_source": "world_pack",
            "termination_count_by_environment": [0] * 32,
            "failed_environment_steps": 0,
            "mean_task_reward": 0.20,
            "mean_reward": 0.18,
            "mean_tracking_score": 0.50,
            "mean_root_height": 0.8,
            "mean_tilt": 0.1,
        }
        candidate = {
            **incumbent,
            "mean_task_reward": 0.28,
            "mean_reward": 0.24,
            "mean_root_height": 0.0,
            "mean_tilt": 1.4,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "candidate")
        self.assertEqual(
            decision["selection_method"],
            "continuous_authored_task_outcome",
        )
        self.assertNotIn("mean tilt increased", decision["regressions"])
        self.assertNotIn(
            "mean root height decreased", decision["regressions"]
        )

    def test_world_pack_failed_step_never_advances(self) -> None:
        incumbent = {
            "task": "velocity",
            "world_source": "urdf",
            "termination_count_by_environment": [0] * 16,
            "failed_environment_steps": 0,
            "mean_task_reward": 0.1,
            "mean_reward": 0.1,
        }
        candidate = {
            **incumbent,
            "failed_environment_steps": 1,
            "mean_task_reward": 0.5,
            "mean_reward": 0.5,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")
        self.assertIn(
            "candidate has failed environment steps",
            decision["regressions"],
        )

    def test_world_pack_reward_sign_change_cannot_buy_total_failure(self) -> None:
        incumbent = {
            "task": "velocity",
            "world_source": "world_pack",
            "termination_count_by_environment": [0] * 16,
            "failed_environment_steps": 0,
            "mean_task_reward": -1.0,
            "mean_reward": -1.0,
            "mean_tracking_score": 0.0,
        }
        candidate = {
            **incumbent,
            "termination_count_by_environment": [1] * 16,
            "termination_count": 16,
            "mean_task_reward": 1.0,
            "mean_reward": 1.0,
            "mean_tracking_score": 1.0,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")
        self.assertLessEqual(decision["selection_score"], 0.0)

    def test_world_pack_checkpoints_compare_to_one_incumbent(self) -> None:
        incumbent = {
            "task": "velocity",
            "world_source": "world_pack",
            "termination_count_by_environment": [0] * 16,
            "failed_environment_steps": 0,
            "mean_task_reward": 0.2,
            "mean_reward": 0.2,
        }
        candidates = {
            "first": {
                **incumbent,
                "mean_task_reward": 0.24,
                "mean_reward": 0.21,
            },
            "best": {
                **incumbent,
                "mean_task_reward": 0.30,
                "mean_reward": 0.27,
            },
        }
        champion, comparisons = select_candidate_champion(
            incumbent, candidates
        )
        self.assertEqual(champion, "best")
        self.assertEqual(set(comparisons), set(candidates))

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

    def test_internal_g1_get_up_task_id_uses_get_up_metrics(self) -> None:
        incumbent = {
            "task": "unitree_g1_supine_get_up_discovery",
            "termination_count": 80,
            "termination_count_by_environment": [0] * 20 + [1] * 80,
            "failed_environment_steps": 0,
            "mean_tilt": 0.2,
            "squat_cycle_completed_environment_rate": 0.10,
        }
        candidate = {
            **incumbent,
            "termination_count": 78,
            "termination_count_by_environment": [0] * 22 + [1] * 78,
            "squat_cycle_completed_environment_rate": 0.12,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["task"], "supine-get-up")
        self.assertEqual(decision["selected"], "candidate")

    def test_developmental_recovery_uses_recovery_metrics(self) -> None:
        incumbent = {
            "task": "unitree_g1_developmental_recovery",
            "termination_count": 0,
            "termination_count_by_environment": [0] * 64,
            "failed_environment_steps": 0,
            "mean_tilt": 1.2,
            "squat_cycle_completed_environment_rate": 0.0,
            "bilateral_support_step_rate": 0.80,
        }
        candidate = {
            **incumbent,
            "squat_cycle_completed_environment_rate": 0.02,
            "bilateral_support_step_rate": 0.82,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["task"], "developmental-recovery")
        self.assertEqual(decision["selected"], "candidate")
        self.assertIn(
            "completed squat-cycle rate increased",
            decision["improvements"],
        )

    def test_developmental_phase_regression_vetoes_support_tradeoff(self) -> None:
        incumbent = {
            "task": "unitree_g1_developmental_recovery",
            "termination_count": 0,
            "termination_count_by_environment": [0] * 64,
            "failed_environment_steps": 0,
            "mean_tilt": 1.2,
            "squat_cycle_completed_environment_rate": 0.0,
            "recovery_phase_rates": {
                "brace": 0.56,
                "foot_support": 0.43,
            },
        }
        candidate = {
            **incumbent,
            "recovery_phase_rates": {
                "brace": 0.43,
                "foot_support": 0.49,
            },
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")
        self.assertIn("brace rate decreased", decision["regressions"])
        self.assertIn(
            "foot_support rate increased",
            decision["improvements"],
        )

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
        authority_index = len(arguments) - 1 - arguments[::-1].index(
            "--interaction-student-authority"
        )
        self.assertEqual(arguments[authority_index + 1], "0")
        self.assertNotIn("--interaction-reset-phase-fraction", arguments)
        environment_index = len(arguments) - 1 - arguments[::-1].index(
            "--envs"
        )
        seed_index = len(arguments) - 1 - arguments[::-1].index("--seed")
        self.assertEqual(arguments[environment_index + 1], "256")
        self.assertEqual(arguments[seed_index + 1], "42")
        steps_index = len(arguments) - 1 - arguments[::-1].index("--steps")
        self.assertEqual(arguments[steps_index + 1], "103")

    def test_generic_sensor_contract_is_preserved_for_evaluation(self) -> None:
        arguments = evaluation_arguments(
            [
                "--world-pack",
                "workcell.mrworld",
                "--task-pack",
                "grasp.taskpack",
                "--robot-actuator-pack",
                "grasp.actuatorpack",
                "--sensor-program-pack",
                "grasp.sensorpack",
                "--reality-program-pack",
                "grasp.realitypack",
                "--visual-observation-config",
                "cameras.json",
                "--visual-environment-pack",
                "studio.mrenv",
                "--envs",
                "64",
            ],
            policy_pack=Path("candidate.policypack"),
            metallib=Path("MetalRobo.metallib"),
            state_trace=Path("candidate.tsv"),
            maximum_environments=32,
            held_out_seed=17,
        )
        for option, value in (
            ("--world-pack", "workcell.mrworld"),
            ("--task-pack", "grasp.taskpack"),
            ("--robot-actuator-pack", "grasp.actuatorpack"),
            ("--sensor-program-pack", "grasp.sensorpack"),
            ("--reality-program-pack", "grasp.realitypack"),
            ("--visual-observation-config", "cameras.json"),
            ("--visual-environment-pack", "studio.mrenv"),
        ):
            index = arguments.index(option)
            self.assertEqual(arguments[index + 1], value)

    def test_birdflow_dove_source_is_preserved_for_evaluation(self) -> None:
        arguments = evaluation_arguments(
            ["--birdflow-dove", "--envs", "8", "--steps", "8"],
            policy_pack=Path("candidate.policypack"),
            metallib=Path("MetalRobo.metallib"),
            state_trace=Path("candidate.state.tsv"),
            maximum_environments=8,
            held_out_seed=7,
        )
        self.assertIn("--birdflow-dove", arguments)

    def test_birdflow_liftoff_tracking_floor_blocks_deployment(self) -> None:
        incumbent = {
            "task": "birdflow_american_crow_standing_to_flight",
            "maximum_sampled_difficulty_band": 2,
            "termination_count": 1,
            "timeout_count": 1,
            "height_or_tilt_termination_count": 0,
            "termination_count_by_environment": [1],
            "failed_environment_steps": 0,
            "mean_tracking_score": 0.49,
            "mean_tilt": 0.46,
            "outcomes": {
                "liftoff": {"mean": 0.8, "direction": 1},
                "push_off": {"mean": 0.1, "direction": 1},
            },
        }
        candidate = {
            **incumbent,
            "mean_tracking_score": 0.69,
            "mean_tilt": 0.36,
            "outcomes": {
                "liftoff": {"mean": 0.9, "direction": 1},
                "push_off": {"mean": 0.2, "direction": 1},
            },
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")
        self.assertIn(
            "standing-to-flight liftoff tracking is below the authored success threshold",
            decision["regressions"],
        )

    def test_adult_selection_isolated_to_highest_training_band(self) -> None:
        arguments = evaluation_arguments(
            [
                "--task",
                "adult-locomotion",
                "--minimum-difficulty-band",
                "2",
                "--maximum-difficulty-band",
                "3",
                "--envs",
                "4096",
                "--steps",
                "32",
            ],
            policy_pack=Path("candidate.policypack"),
            metallib=Path("MetalRobo.metallib"),
            state_trace=Path("candidate.tsv"),
            maximum_environments=512,
            held_out_seed=42,
        )
        minimum_index = len(arguments) - 1 - arguments[::-1].index(
            "--minimum-difficulty-band"
        )
        maximum_index = len(arguments) - 1 - arguments[::-1].index(
            "--maximum-difficulty-band"
        )
        self.assertEqual(arguments[minimum_index + 1], "3")
        self.assertEqual(arguments[maximum_index + 1], "3")

    def test_adult_previous_band_selection_override_is_exact(self) -> None:
        arguments = evaluation_arguments(
            [
                "--task",
                "adult-locomotion",
                "--minimum-difficulty-band",
                "2",
                "--maximum-difficulty-band",
                "3",
            ],
            policy_pack=Path("candidate.policypack"),
            metallib=Path("MetalRobo.metallib"),
            state_trace=Path("candidate.tsv"),
            maximum_environments=512,
            held_out_seed=42,
            evaluation_minimum_band=2,
            evaluation_maximum_band=2,
        )
        minimum_index = len(arguments) - 1 - arguments[::-1].index(
            "--minimum-difficulty-band"
        )
        maximum_index = len(arguments) - 1 - arguments[::-1].index(
            "--maximum-difficulty-band"
        )
        self.assertEqual(arguments[minimum_index + 1], "2")
        self.assertEqual(arguments[maximum_index + 1], "2")

    def test_crow_selection_isolated_to_newest_training_band(self) -> None:
        arguments = evaluation_arguments(
            [
                "--birdflow-american-crow",
                "--minimum-difficulty-band",
                "1",
                "--maximum-difficulty-band",
                "2",
                "--envs",
                "128",
                "--steps",
                "64",
            ],
            policy_pack=Path("candidate.policypack"),
            metallib=Path("MetalRobo.metallib"),
            state_trace=Path("candidate.tsv"),
            maximum_environments=64,
            held_out_seed=42,
        )
        minimum_index = len(arguments) - 1 - arguments[::-1].index(
            "--minimum-difficulty-band"
        )
        maximum_index = len(arguments) - 1 - arguments[::-1].index(
            "--maximum-difficulty-band"
        )
        self.assertEqual(arguments[minimum_index + 1], "2")
        self.assertEqual(arguments[maximum_index + 1], "2")

    def test_crow_previous_band_override_is_exact(self) -> None:
        arguments = evaluation_arguments(
            [
                "--birdflow-american-crow",
                "--minimum-difficulty-band",
                "1",
                "--maximum-difficulty-band",
                "2",
            ],
            policy_pack=Path("candidate.policypack"),
            metallib=Path("MetalRobo.metallib"),
            state_trace=Path("candidate.tsv"),
            maximum_environments=64,
            held_out_seed=42,
            evaluation_minimum_band=1,
            evaluation_maximum_band=1,
        )
        minimum_index = len(arguments) - 1 - arguments[::-1].index(
            "--minimum-difficulty-band"
        )
        maximum_index = len(arguments) - 1 - arguments[::-1].index(
            "--maximum-difficulty-band"
        )
        self.assertEqual(arguments[minimum_index + 1], "1")
        self.assertEqual(arguments[maximum_index + 1], "1")

    def test_crow_journey_selection_targets_terminal_journey_band(self) -> None:
        arguments = evaluation_arguments(
            [
                "--birdflow-american-crow-journey",
                "--birdflow-journey-teacher",
                "--minimum-difficulty-band",
                "0",
                "--maximum-difficulty-band",
                "4",
            ],
            policy_pack=Path("candidate.policypack"),
            metallib=Path("MetalRobo.metallib"),
            state_trace=Path("candidate.tsv"),
            maximum_environments=8,
            held_out_seed=42,
        )
        self.assertIn("--birdflow-american-crow-journey", arguments)
        self.assertNotIn("--birdflow-journey-teacher", arguments)
        minimum_index = len(arguments) - 1 - arguments[::-1].index(
            "--minimum-difficulty-band"
        )
        maximum_index = len(arguments) - 1 - arguments[::-1].index(
            "--maximum-difficulty-band"
        )
        self.assertEqual(arguments[minimum_index + 1], "4")
        self.assertEqual(arguments[maximum_index + 1], "4")

    def test_crow_takeoff_cruise_requires_absolute_physical_gate(self) -> None:
        incumbent = {
            "task": "birdflow_american_crow_journey_v7",
            "maximum_sampled_difficulty_band": 4,
            "termination_count_by_environment": [0] * 8,
            "failed_environment_steps": 0,
            "maximum_root_height": 0.30,
            "maximum_tilt": 0.10,
            "mean_tracking_score": 0.60,
            "mean_tilt": 0.10,
            "outcomes": {
                "figure_eight_tracking": {"mean": 0.0, "direction": 1},
                "liftoff": {"mean": 0.1, "direction": 1},
                "walking_contact": {"mean": 0.0, "direction": 1},
                "ground_support": {"mean": 0.1, "direction": 1},
                "tracking": {"mean": 0.6, "direction": 1},
            },
        }
        candidate = {
            **incumbent,
            "maximum_root_height": 0.50,
            "mean_tracking_score": 0.64,
            "outcomes": {
                **incumbent["outcomes"],
                "liftoff": {"mean": 0.2, "direction": 1},
            },
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")
        self.assertIn(
            "takeoff-cruise candidate did not reach the 0.55 m liftoff gate",
            decision["regressions"],
        )

    def test_crow_previous_band_regression_blocks_flight_progress(self) -> None:
        current_incumbent = {
            "task": "birdflow_american_crow_standing_to_flight",
            "maximum_sampled_difficulty_band": 2,
            "termination_count_by_environment": [0] * 8,
            "failed_environment_steps": 0,
            "mean_tracking_score": 0.70,
            "mean_tilt": 0.10,
            "outcomes": {
                "liftoff": {"mean": 0.40, "direction": 1},
                "push_off": {"mean": 0.10, "direction": 1},
            },
        }
        current_candidate = {
            **current_incumbent,
            "mean_tracking_score": 0.75,
            "outcomes": {
                "liftoff": {"mean": 0.50, "direction": 1},
                "push_off": {"mean": 0.20, "direction": 1},
            },
        }
        previous_incumbent = {
            **current_incumbent,
            "maximum_sampled_difficulty_band": 1,
            "mean_tracking_score": 0.60,
            "outcomes": {
                "ground_support": {"mean": 0.80, "direction": 1},
                "walking_contact": {"mean": 0.50, "direction": 1},
            },
        }
        previous_candidate = {
            **previous_incumbent,
            "outcomes": {
                "ground_support": {"mean": 0.60, "direction": 1},
                "walking_contact": {"mean": 0.40, "direction": 1},
            },
        }
        decision = compare_staged_bands(
            current_incumbent,
            current_candidate,
            previous_incumbent,
            previous_candidate,
        )
        self.assertEqual(decision["selected"], "incumbent")
        self.assertIn(
            "previous-band: ground_support decreased",
            decision["regressions"],
        )

    def test_crow_previous_band_tracking_regression_blocks_flight_progress(
        self,
    ) -> None:
        current_incumbent = {
            "task": "birdflow_american_crow_standing_to_flight",
            "maximum_sampled_difficulty_band": 2,
            "termination_count_by_environment": [0] * 8,
            "failed_environment_steps": 0,
            "mean_tracking_score": 0.70,
            "mean_tilt": 0.10,
            "outcomes": {
                "liftoff": {"mean": 0.40, "direction": 1},
                "push_off": {"mean": 0.10, "direction": 1},
            },
        }
        current_candidate = {
            **current_incumbent,
            "mean_tracking_score": 0.75,
            "outcomes": {
                "liftoff": {"mean": 0.50, "direction": 1},
                "push_off": {"mean": 0.20, "direction": 1},
            },
        }
        previous_incumbent = {
            **current_incumbent,
            "maximum_sampled_difficulty_band": 1,
            "mean_tracking_score": 0.60,
            "outcomes": {
                "ground_support": {"mean": 0.80, "direction": 1},
                "walking_contact": {"mean": 0.50, "direction": 1},
            },
        }
        previous_candidate = {
            **previous_incumbent,
            "mean_tracking_score": 0.59,
        }
        decision = compare_staged_bands(
            current_incumbent,
            current_candidate,
            previous_incumbent,
            previous_candidate,
        )
        self.assertEqual(decision["selected"], "incumbent")
        self.assertIn(
            "previous-band: tracking score decreased",
            decision["regressions"],
        )

    def test_crow_journey_protects_every_prior_band(self) -> None:
        current = {
            "task": "birdflow_american_crow_journey_v7",
            "maximum_sampled_difficulty_band": 10,
            "termination_count_by_environment": [0] * 8,
            "failed_environment_steps": 0,
            "maximum_root_height": 0.85,
            "maximum_tilt": 0.40,
            "mean_tilt": 0.10,
            "mean_tracking_score": 0.72,
            "outcomes": {
                "figure_eight_tracking": {"mean": 0.5, "direction": 1},
                "liftoff": {"mean": 0.5, "direction": 1},
            },
        }
        protected_incumbent = {
            **current,
            "maximum_sampled_difficulty_band": 0,
            "mean_tracking_score": 0.95,
            "outcomes": {
                "ground_support": {"mean": 0.9, "direction": 1},
            },
        }
        protected_candidate = {
            **protected_incumbent,
            "mean_tracking_score": 0.80,
            "outcomes": {
                "ground_support": {"mean": 0.7, "direction": 1},
            },
        }
        decision = compare_protected_bands(
            {**current, "mean_tracking_score": 0.70},
            current,
            {0: (protected_incumbent, protected_candidate)},
        )
        self.assertEqual(decision["selected"], "incumbent")
        self.assertIn(
            "protected-band-0: tracking below milestone floor 0.95",
            decision["regressions"],
        )

    def test_crow_journey_allows_safe_relative_tradeoff(self) -> None:
        full = {
            "task": "birdflow_american_crow_journey_v7",
            "maximum_sampled_difficulty_band": 10,
            "termination_count": 8,
            "timeout_count": 8,
            "termination_count_by_environment": [1] * 8,
            "failed_environment_steps": 0,
            "maximum_root_height": 0.85,
            "maximum_tilt": 0.40,
            "mean_tilt": 0.10,
            "mean_tracking_score": 0.70,
            "outcomes": {
                "tracking": {"mean": 0.70, "direction": 1},
                "liftoff": {"mean": 0.5, "direction": 1},
                "ground_support": {"mean": 0.5, "direction": 1},
            },
        }
        protected_incumbent = {
            **full,
            "maximum_sampled_difficulty_band": 2,
            "maximum_root_height": 0.90,
            "mean_tracking_score": 0.80,
            "outcomes": {
                "forward_flight_tracking": {"mean": 0.8, "direction": 1},
                "liftoff": {"mean": 0.8, "direction": 1},
                "push_off": {"mean": 0.2, "direction": 1},
            },
        }
        protected_candidate = {
            **protected_incumbent,
            "maximum_root_height": 0.75,
            "mean_tracking_score": 0.72,
            "outcomes": {
                "forward_flight_tracking": {"mean": 0.7, "direction": 1},
                "liftoff": {"mean": 0.7, "direction": 1},
                "push_off": {"mean": 0.1, "direction": 1},
            },
        }
        decision = compare_protected_bands(
            full,
            {**full, "mean_tracking_score": 0.71},
            {2: (protected_incumbent, protected_candidate)},
        )
        self.assertEqual(decision["selected"], "candidate")
        self.assertEqual(decision["regressions"], [])
        self.assertIn(
            "forward_flight_tracking decreased",
            decision["protected_band_comparisons"]["2"][
                "relative_regressions"
            ],
        )

    def test_checkpoint_comparisons_are_json_serializable(self) -> None:
        incumbent = {
            "task": "adult-locomotion",
            "forward_progress_available": True,
            "termination_count_by_environment": [0],
            "failed_environment_steps": 0,
            "mean_peak_forward_progress_m": 0.5,
            "mean_final_forward_progress_m": 0.4,
            "mean_tracking_score": 0.4,
            "mean_root_height": 0.70,
            "mean_tilt": 0.30,
        }
        candidate = {
            **incumbent,
            "mean_peak_forward_progress_m": 0.6,
            "mean_final_forward_progress_m": 0.5,
            "mean_tracking_score": 0.5,
        }
        _, comparisons = select_candidate_champion(
            incumbent, {"candidate": candidate}
        )
        decision = dict(comparisons["candidate"])
        decision["checkpoint_comparisons"] = comparisons
        json.dumps(decision)

    def test_matching_evidence_contract_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "evaluator.sh"
            counter = root / "counter"
            state_trace = root / "candidate.state.tsv"
            evidence = root / "candidate.evidence.json"
            evaluator.write_text(
                "#!/bin/sh\n"
                f"count=0; test -f '{counter}' && count=$(cat '{counter}')\n"
                "count=$((count + 1))\n"
                f"printf '%s' \"$count\" > '{counter}'\n"
                "printf '# step nq=1 scene_bodies=0 scene_stride=13 "
                "timestep=0.02 environment=0\\n1\\t0\\n' "
                f"> '{state_trace}'\n"
                "printf '%s\\n' '{\"task\":\"adult-locomotion\",\"failed_environment_steps\":0}'\n",
                encoding="utf-8",
            )
            evaluator.chmod(evaluator.stat().st_mode | 0o111)
            metallib = root / "MetalRobo.metallib"
            policy = root / "candidate.policypack"
            metallib.write_bytes(b"metallib")
            policy.write_bytes(b"policy")
            arguments = [
                "--task",
                "adult-locomotion",
                "--envs",
                "2",
                "--steps",
                "1",
                "--seed",
                "7",
                "--metallib",
                str(metallib),
                "--policy-pack",
                str(policy),
                "--state-trace",
                str(state_trace),
            ]
            first = _evaluate(evaluator, arguments, evidence)
            second = _evaluate(evaluator, arguments, evidence)
            self.assertEqual(first, second)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")
            metadata = json.loads(
                evidence.with_name(f"{evidence.name}.meta.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                metadata["schema"], "numi.policy-evidence-cache.v2"
            )
            self.assertEqual(metadata["state_trace"]["row_count"], 1)
            self.assertEqual(metadata["state_trace"]["terminal_step"], 1)

            state_trace.write_text(
                "# step nq=1 scene_bodies=0 scene_stride=13 "
                "timestep=0.02 environment=1\n1\t0\n",
                encoding="utf-8",
            )
            regenerated = _evaluate(evaluator, arguments, evidence)
            self.assertEqual(regenerated, first)
            self.assertEqual(counter.read_text(encoding="utf-8"), "2")

            state_trace.write_text(
                "# step nq=1 scene_bodies=0 scene_stride=13 "
                "timestep=0.02 environment=0\n",
                encoding="utf-8",
            )
            regenerated = _evaluate(evaluator, arguments, evidence)
            self.assertEqual(regenerated, first)
            self.assertEqual(counter.read_text(encoding="utf-8"), "3")

            evidence.write_text("{}\n", encoding="utf-8")
            regenerated = _evaluate(evaluator, arguments, evidence)
            self.assertEqual(regenerated, first)
            self.assertEqual(counter.read_text(encoding="utf-8"), "4")

    def test_single_band_adult_training_still_protects_previous_rung(self) -> None:
        current, previous = _adult_evaluation_bands(
            [
                "--task",
                "adult-locomotion",
                "--minimum-difficulty-band",
                "2",
                "--maximum-difficulty-band",
                "2",
            ]
        )
        self.assertEqual((current, previous), (2, 1))
        self.assertEqual(
            _adult_evaluation_bands(
                [
                    "--task",
                    "adult-locomotion",
                    "--maximum-difficulty-band",
                    "0",
                ]
            ),
            (0, None),
        )

    def test_crow_curriculum_protects_earlier_walking_rung(self) -> None:
        current, previous = _curriculum_evaluation_bands(
            [
                "--birdflow-american-crow",
                "--minimum-difficulty-band",
                "1",
                "--maximum-difficulty-band",
                "2",
            ]
        )
        self.assertEqual((current, previous), (2, 1))

    def test_crow_journey_curriculum_protects_prejourney_flight_rung(self) -> None:
        current, previous = _curriculum_evaluation_bands(
            [
                "--birdflow-american-crow-journey",
                "--minimum-difficulty-band",
                "0",
                "--maximum-difficulty-band",
                "4",
            ]
        )
        self.assertEqual((current, previous), (4, 3))

    def test_visual_crow_selection_retains_variant(self) -> None:
        arguments = evaluation_arguments(
            [
                "--birdflow-american-crow-journey",
                "--birdflow-journey-variant", "v9-visual-neural",
                "--visual-observation-config", "/tmp/crow-visual.json",
                "--updates", "10",
            ],
            policy_pack=Path("/tmp/policy.policypack"),
            metallib=Path("/tmp/MetalRobo.metallib"),
            state_trace=Path("/tmp/crow.tsv"),
            maximum_environments=2,
            held_out_seed=7,
            evaluation_steps=4,
        )
        self.assertIn("--birdflow-journey-variant", arguments)
        self.assertIn("v9-visual-neural", arguments)

    def test_neural_crow_approach_requires_shadow_envelope_clearance(self) -> None:
        record = {
            "task": "birdflow_american_crow_journey_v8_neural",
            "failed_environment_steps": 0,
            "termination_count": 0,
            "timeout_count": 0,
            "mean_tracking_score": 0.8,
            "mean_tilt": 0.1,
            "maximum_tilt": 0.2,
            "maximum_root_height": 0.8,
            "outcomes": {
                "approach_pitch_warning_fraction": {
                    "mean": 0.06, "direction": 2,
                },
                "approach_pitch_full_envelope_fraction": {
                    "mean": 0.0, "direction": 2,
                },
            },
        }
        self.assertIn(
            "neural approach warning-envelope occupancy exceeds 0.05",
            _crow_journey_contract_regressions(record, 7),
        )
        record["outcomes"]["approach_pitch_warning_fraction"]["mean"] = 0.01
        self.assertEqual(_crow_journey_contract_regressions(record, 7), [])

    def test_crow_band_zero_uses_absolute_contract_without_protected_bands(self) -> None:
        record = {
            "task": "birdflow_american_crow_journey_v8_neural",
            "failed_environment_steps": 0,
            "termination_count": 512,
            "timeout_count": 512,
            "mean_tracking_score": 1.0,
            "mean_tilt": 0.001,
            "maximum_tilt": 0.01,
            "maximum_root_height": 0.19,
            "minimum_sampled_difficulty_band": 0,
            "maximum_sampled_difficulty_band": 0,
        }
        decision = compare_protected_bands(record, record, {})
        self.assertEqual(decision["selected"], "candidate")
        self.assertTrue(decision["candidate_advanced_deployment"])
        self.assertEqual(decision["regressions"], [])
        self.assertEqual(
            decision["selection_method"],
            "birdflow_crow_journey_absolute_protected_contract",
        )

    def test_crow_current_contract_failure_is_sufficient_to_reject(self) -> None:
        incumbent = {
            "task": "birdflow_american_crow_journey_v8_neural",
            "failed_environment_steps": 0,
            "termination_count": 512,
            "timeout_count": 512,
            "mean_tracking_score": 0.92,
            "mean_tilt": 0.01,
            "maximum_tilt": 0.02,
            "maximum_root_height": 0.19,
            "minimum_sampled_difficulty_band": 1,
            "maximum_sampled_difficulty_band": 1,
        }
        candidate = {
            **incumbent,
            "mean_tracking_score": 0.91,
            "mean_tilt": 0.36,
        }
        failures = _crow_journey_contract_regressions(candidate, 1)
        self.assertIn("mean tilt exceeds 0.35 rad", failures)
        decision = compare_protected_bands(incumbent, candidate, {})
        self.assertEqual(decision["selected"], "incumbent")
        self.assertFalse(decision["candidate_advanced_deployment"])
        self.assertIn("mean tilt exceeds 0.35 rad", decision["regressions"])

    def test_crow_absolute_failure_cannot_win_on_relative_score(self) -> None:
        incumbent = {
            "task": "birdflow_american_crow_journey_v8_neural",
            "failed_environment_steps": 0,
            "termination_count": 512,
            "timeout_count": 0,
            "mean_tracking_score": 0.64,
            "mean_tilt": 0.08,
            "maximum_tilt": 0.70,
            "maximum_root_height": 0.80,
            "minimum_sampled_difficulty_band": 10,
            "maximum_sampled_difficulty_band": 10,
            "outcomes": {
                "approach_pitch_warning_fraction": {
                    "mean": 0.20, "direction": 2,
                },
                "approach_pitch_full_envelope_fraction": {
                    "mean": 0.10, "direction": 2,
                },
            },
        }
        candidate = {
            **incumbent,
            "termination_count": 512,
            "timeout_count": 512,
            "mean_tracking_score": 0.67,
            "mean_tilt": 0.06,
            "maximum_tilt": 0.60,
            "outcomes": {
                "approach_pitch_warning_fraction": {
                    "mean": 0.01, "direction": 2,
                },
                "approach_pitch_full_envelope_fraction": {
                    "mean": 0.008, "direction": 2,
                },
            },
        }
        decision = compare_protected_bands(incumbent, candidate, {})
        self.assertEqual(decision["selected"], "incumbent")
        self.assertFalse(decision["candidate_advanced_deployment"])
        self.assertIn(
            "neural approach full-envelope occupancy is nonzero",
            decision["regressions"],
        )

    def test_crow_selector_skips_protected_replay_after_current_failure(self) -> None:
        record = {
            "task": "birdflow_american_crow_journey_v8_neural",
            "failed_environment_steps": 0,
            "termination_count": 512,
            "timeout_count": 512,
            "mean_tracking_score": 0.92,
            "mean_tilt": 0.01,
            "maximum_tilt": 0.02,
            "maximum_root_height": 0.19,
            "minimum_sampled_difficulty_band": 1,
            "maximum_sampled_difficulty_band": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent = root / "incumbent.policypack"
            candidate = root / "candidate.policypack"
            deployment = root / "deployment.policypack"
            evidence = root / "selection"
            evaluator = root / "evaluator"
            metallib = root / "MetalRobo.metallib"
            for path in (incumbent, candidate, evaluator, metallib):
                path.write_text(path.name, encoding="utf-8")

            evaluated: list[str] = []

            def fake_evaluate(
                _evaluator: Path,
                _arguments: list[str],
                evidence_path: Path,
            ) -> dict[str, object]:
                evaluated.append(evidence_path.name)
                if evidence_path.name == "candidate.evidence.json":
                    return {**record, "mean_tilt": 0.36}
                if evidence_path.name.startswith("candidate.protected-band"):
                    self.fail("failed current candidate reached protected replay")
                return dict(record)

            arguments = [
                "policy_selection.py",
                "--evaluator", str(evaluator),
                "--metallib", str(metallib),
                "--incumbent", str(incumbent),
                "--candidate", str(candidate),
                "--deployment", str(deployment),
                "--evidence-directory", str(evidence),
                "--maximum-environments", "512",
                "--evaluation-steps", "1600",
                "--",
                "--birdflow-american-crow-journey",
                "--minimum-difficulty-band", "1",
                "--maximum-difficulty-band", "1",
            ]
            with patch.object(sys, "argv", arguments), patch.object(
                policy_selection, "_evaluate", side_effect=fake_evaluate
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(policy_selection.main(), 0)

            self.assertEqual(
                evaluated,
                [
                    "incumbent.evidence.json",
                    "incumbent.protected-band-0.evidence.json",
                    "candidate.evidence.json",
                ],
            )
            decision = json.loads(
                (evidence / "selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(decision["selected"], "incumbent")
            self.assertEqual(decision["protected_bands_skipped"], [0])


if __name__ == "__main__":
    unittest.main()
