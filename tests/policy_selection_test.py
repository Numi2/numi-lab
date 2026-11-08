#!/usr/bin/env python3

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from metalrobo.policy_selection import (  # noqa: E402
    _adult_evaluation_bands,
    _cache_reusable,
    _evaluation_contract,
    _evaluate,
    compare_adult_bands,
    compare_evidence,
    evaluation_arguments,
    select_candidate_champion,
)


class PolicySelectionTest(unittest.TestCase):
    def test_dove_drop_recovery_ignores_planar_drift(self) -> None:
        incumbent = {
            "task": "deetjen_f03_robot.fatal_drop_recovery",
            "world_source": "measured_dove",
            "termination_count_by_environment": [1, 0, 0, 0],
            "height_or_tilt_termination_count": 1,
            "clean_horizon_environment_rate": 0.75,
            "failed_environment_steps": 0,
            "mean_reward": -0.16,
            "mean_root_height": 4.3,
            "mean_tilt": 1.12,
            "mean_peak_forward_progress_m": 3.0,
        }
        candidate = {
            **incumbent,
            "termination_count_by_environment": [0, 0, 0, 0],
            "height_or_tilt_termination_count": 0,
            "clean_horizon_environment_rate": 1.0,
            "mean_reward": -0.05,
            "mean_root_height": 4.9,
            "mean_tilt": 0.88,
            "mean_peak_forward_progress_m": 1.8,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "candidate")
        self.assertEqual(
            decision["selection_method"],
            "dove_flight_recovery_comparison",
        )
        self.assertNotIn(
            "mean peak forward progress decreased",
            decision["regressions"],
        )

    def test_dove_ground_contact_counts_as_physical_failure(self) -> None:
        incumbent = {
            "task": "deetjen_f03_robot.fatal_drop_recovery",
            "world_source": "measured_dove",
            "termination_count_by_environment": [0, 0, 0, 0],
            "termination_count": 0,
            "timeout_count": 0,
            "height_or_tilt_termination_count": 0,
            "clean_horizon_environment_rate": 1.0,
            "failed_environment_steps": 0,
            "mean_reward": 0.0,
            "mean_root_height": 5.0,
            "mean_tilt": 0.5,
        }
        candidate = {
            **incumbent,
            "termination_count_by_environment": [1, 0, 0, 0],
            "termination_count": 1,
            "termination_reason_counts": {"3": 1},
            "clean_horizon_environment_rate": 0.75,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")
        self.assertEqual(
            decision["metrics"]["candidate_physical_failure_rate"],
            0.25,
        )

    def test_dove_reason_count_timeouts_are_not_physical_failures(self) -> None:
        incumbent = {
            "task": "deetjen_f03_robot.fatal_drop_recovery",
            "world_source": "measured_dove",
            "termination_count_by_environment": [1, 0, 0, 0],
            "termination_count": 1,
            "termination_reason_counts": {"4": 1},
            "clean_horizon_environment_rate": 0.75,
            "failed_environment_steps": 0,
            "mean_reward": -0.1,
            "mean_root_height": 5.0,
            "mean_tilt": 0.5,
        }
        candidate = {
            **incumbent,
            "mean_reward": -0.09,
            "mean_root_height": 5.1,
            "mean_tilt": 0.49,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(
            decision["metrics"]["candidate_physical_failure_rate"],
            0.0,
        )
        self.assertEqual(decision["selected"], "candidate")

    def test_dove_authored_forward_tracking_guides_selection(self) -> None:
        incumbent = {
            "task": "deetjen_f03_robot.fatal_drop_recovery",
            "world_source": "measured_dove",
            "termination_count_by_environment": [0] * 16,
            "termination_count": 0,
            "clean_horizon_environment_rate": 1.0,
            "failed_environment_steps": 0,
            "mean_reward": 0.1,
            "mean_root_height": 22.0,
            "mean_tilt": 0.5,
            "outcomes": {
                "forward_speed_tracking": {
                    "mean": 0.20,
                    "direction": 1,
                }
            },
        }
        candidate = {
            **incumbent,
            "outcomes": {
                "forward_speed_tracking": {
                    "mean": 0.30,
                    "direction": 1,
                }
            },
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "candidate")
        self.assertIn(
            "forward-speed tracking increased", decision["improvements"]
        )
        self.assertEqual(
            decision["metrics"]["adult_authored_outcomes"]
            ["candidate_forward_speed_tracking"]["mean"],
            0.30,
        )

    def test_dove_lift_gain_cannot_buy_lost_forward_tracking(self) -> None:
        incumbent = {
            "task": "deetjen_f03_robot.fatal_drop_recovery",
            "world_source": "measured_dove",
            "termination_count_by_environment": [0] * 16,
            "termination_count": 0,
            "clean_horizon_environment_rate": 1.0,
            "failed_environment_steps": 0,
            "mean_reward": 0.1,
            "mean_root_height": 22.0,
            "mean_tilt": 0.5,
            "outcomes": {
                "forward_speed_tracking": {
                    "mean": 0.00110,
                    "direction": 1,
                }
            },
        }
        candidate = {
            **incumbent,
            "mean_reward": 0.2,
            "mean_root_height": 23.0,
            "outcomes": {
                "forward_speed_tracking": {
                    "mean": 0.00100,
                    "direction": 1,
                }
            },
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")

    def test_dove_forward_gain_cannot_buy_lost_heading_tracking(self) -> None:
        incumbent = {
            "task": "deetjen_f03_robot.fatal_drop_recovery",
            "world_source": "measured_dove",
            "termination_count_by_environment": [0] * 16,
            "termination_count": 0,
            "clean_horizon_environment_rate": 1.0,
            "failed_environment_steps": 0,
            "mean_reward": 0.1,
            "mean_root_height": 22.0,
            "mean_tilt": 0.5,
            "outcomes": {
                "forward_speed_tracking": {"mean": 0.0010, "direction": 1},
                "heading_rate_tracking": {"mean": 0.0011, "direction": 1},
            },
        }
        candidate = {
            **incumbent,
            "outcomes": {
                "forward_speed_tracking": {"mean": 0.0012, "direction": 1},
                "heading_rate_tracking": {"mean": 0.0010, "direction": 1},
            },
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["selected"], "incumbent")

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

    def test_generic_physical_regression_cannot_be_bought_with_reward(self) -> None:
        for source in ("world_pack", "urdf"):
            with self.subTest(source=source):
                incumbent = {
                    "task": "grasp",
                    "world_source": source,
                    "termination_count_by_environment": [0] * 16,
                    "termination_count": 0,
                    "failed_environment_steps": 0,
                    "mean_task_reward": 0.1,
                    "mean_reward": 0.1,
                }
                candidate = {
                    **incumbent,
                    "termination_count_by_environment": [1] + [0] * 15,
                    "termination_count": 1,
                    "termination_reason_counts": {"3": 1},
                    "mean_task_reward": 0.2,
                    "mean_reward": 0.2,
                }
                decision = compare_evidence(incumbent, candidate)
                self.assertGreater(decision["selection_score"], 0)
                self.assertIn("termination rate increased", decision["regressions"])
                self.assertEqual(decision["selected"], "incumbent")

    def test_generic_timeout_does_not_block_reward_gain(self) -> None:
        incumbent = {
            "task": "grasp",
            "world_source": "world_pack",
            "termination_count_by_environment": [0] * 16,
            "termination_count": 0,
            "failed_environment_steps": 0,
            "mean_task_reward": 0.1,
            "mean_reward": 0.1,
        }
        candidate = {
            **incumbent,
            "termination_count_by_environment": [1] + [0] * 15,
            "termination_count": 1,
            "termination_reason_counts": {"4": 1},
            "mean_task_reward": 0.2,
            "mean_reward": 0.2,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["metrics"]["candidate_physical_failure_rate"], 0)
        self.assertEqual(decision["selected"], "candidate")

    def test_selector_cli_preserves_incumbent_on_physical_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incumbent, candidate = root / "incumbent.pack", root / "candidate.pack"
            deployment, shader = root / "deployment.pack", root / "shader"
            evaluator, evidence = root / "evaluator.py", root / "evidence"
            incumbent.write_bytes(b"incumbent-policy")
            candidate.write_bytes(b"candidate-policy")
            shader.write_bytes(b"shader")
            evaluator.write_text(
                "#!" + sys.executable + "\n"
                "import json, pathlib, sys\n"
                "args=sys.argv[1:]\n"
                "policy=pathlib.Path(args[args.index('--policy-pack')+1])\n"
                "trace=pathlib.Path(args[args.index('--state-trace')+1])\n"
                "trace.write_text('# step nq=1 scene_bodies=0 scene_stride=13 timestep=0.02 environment=0\\n1\\t0\\n')\n"
                "candidate=policy.name == 'candidate.pack'\n"
                "print(json.dumps({'task':'grasp','world_source':'world_pack',"
                "'termination_count_by_environment':([1]+[0]*15 if candidate else [0]*16),"
                "'termination_count':int(candidate),'termination_reason_counts':({'3':1} if candidate else {}),"
                "'failed_environment_steps':0,'mean_task_reward':(0.2 if candidate else 0.1),"
                "'mean_reward':(0.2 if candidate else 0.1)}))\n",
                encoding="utf-8",
            )
            evaluator.chmod(0o755)
            selector = Path(__file__).resolve().parents[1] / "python/metalrobo/policy_selection.py"
            completed = subprocess.run(
                [sys.executable, str(selector), "--evaluator", str(evaluator),
                 "--metallib", str(shader), "--incumbent", str(incumbent),
                 "--candidate", str(candidate), "--deployment", str(deployment),
                 "--evidence-directory", str(evidence), "--maximum-environments", "16",
                 "--held-out-seed", "42", "--evaluation-steps", "1",
                 "--", "--task", "grasp", "--envs", "16", "--steps", "1"],
                text=True, capture_output=True, check=True,
            )
            decision = json.loads(completed.stdout)
            self.assertEqual(decision["selected"], "incumbent")
            self.assertIn("termination rate increased", decision["regressions"])
            self.assertEqual(deployment.read_bytes(), incumbent.read_bytes())
            self.assertEqual(candidate.read_bytes(), b"candidate-policy")
            self.assertTrue((evidence / "candidate.evidence.json.meta.json").is_file())

    def test_generic_success_termination_is_not_a_physical_failure(self) -> None:
        incumbent = {
            "task": "foraging",
            "world_source": "world_pack",
            "termination_count_by_environment": [0] * 16,
            "termination_count": 0,
            "failed_environment_steps": 0,
            "mean_task_reward": 0.1,
            "mean_reward": 0.1,
        }
        candidate = {
            **incumbent,
            "termination_count_by_environment": [1] + [0] * 15,
            "termination_count": 1,
            "termination_reason_counts": {"7": 1},
            "mean_task_reward": 0.2,
            "mean_reward": 0.2,
        }
        decision = compare_evidence(incumbent, candidate)
        self.assertEqual(decision["metrics"]["candidate_physical_failure_rate"], 0)
        self.assertEqual(decision["selected"], "candidate")

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
        self.assertNotIn("--interaction-reset-only", arguments)
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

    def test_interaction_push_contract_is_preserved_for_evaluation(self) -> None:
        arguments = evaluation_arguments(
            [
                "--task", "velocity",
                "--interaction-pack", "teacher.interactionpack",
                "--interaction-clip", "raise-right-hand",
                "--interaction-student-authority", "0.1",
                "--interaction-push-maximum-velocity", "0.2",
                "--interaction-push-interval-seconds", "0.4",
            ],
            policy_pack=Path("candidate.policypack"),
            metallib=Path("MetalRobo.metallib"),
            state_trace=Path("candidate.tsv"),
            maximum_environments=16,
            held_out_seed=42,
            evaluation_steps=40,
        )
        velocity = arguments.index("--interaction-push-maximum-velocity")
        interval = arguments.index("--interaction-push-interval-seconds")
        self.assertEqual(arguments[velocity + 1], "0.2")
        self.assertEqual(arguments[interval + 1], "0.4")

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

    def test_measured_surface_manifests_survive_projection(self) -> None:
        for option in ("--dove-manifest", "--numifly-wing-manifest"):
            with self.subTest(option=option):
                arguments = evaluation_arguments(
                    [option, "authored.json"], policy_pack=Path("policy"),
                    metallib=Path("shader"), state_trace=Path("trace"),
                    maximum_environments=1, held_out_seed=4)
                self.assertEqual(arguments[arguments.index(option) + 1], "authored.json")

    def test_measured_dove_manifest_is_preserved_for_evaluation(self) -> None:
        arguments = evaluation_arguments(
            [
                "--task",
                "dove-drop-recovery",
                "--dove-manifest",
                "deetjen/manifest.json",
                "--envs",
                "64",
            ],
            policy_pack=Path("candidate.policypack"),
            metallib=Path("MetalRobo.metallib"),
            state_trace=Path("candidate.tsv"),
            maximum_environments=32,
            held_out_seed=17,
        )
        index = arguments.index("--dove-manifest")
        self.assertEqual(arguments[index + 1], "deetjen/manifest.json")

    def test_dove_selection_isolated_to_highest_training_band(self) -> None:
        arguments = evaluation_arguments(
            [
                "--task",
                "dove-drop-recovery",
                "--dove-manifest",
                "deetjen/manifest.json",
                "--minimum-difficulty-band",
                "0",
                "--maximum-difficulty-band",
                "4",
            ],
            policy_pack=Path("candidate.policypack"),
            metallib=Path("MetalRobo.metallib"),
            state_trace=Path("candidate.tsv"),
            maximum_environments=256,
            held_out_seed=17,
        )
        minimum_index = len(arguments) - 1 - arguments[::-1].index(
            "--minimum-difficulty-band"
        )
        maximum_index = len(arguments) - 1 - arguments[::-1].index(
            "--maximum-difficulty-band"
        )
        self.assertEqual(arguments[minimum_index + 1], "4")
        self.assertEqual(arguments[maximum_index + 1], "4")

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
            self.assertEqual(metadata["policy_sha256"], hashlib.sha256(policy.read_bytes()).hexdigest())
            metadata.pop("policy_sha256")
            metadata_path = evidence.with_name(f"{evidence.name}.meta.json")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(_evaluate(evaluator, arguments, evidence), first)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["policy_sha256"], hashlib.sha256(policy.read_bytes()).hexdigest())
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

    def test_authored_input_bytes_invalidate_contract_and_missing_fails(self) -> None:
        # These are the direct self-contained runtime files accepted by the selector.
        options = ("--metallib", "--policy-pack", "--interaction-pack", "--world-pack",
                   "--task-pack", "--robot-actuator-pack", "--sensor-program-pack",
                   "--reality-program-pack", "--srdf", "--visual-environment-pack")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "evaluator"
            evaluator.write_bytes(b"evaluator")
            for option in options:
                with self.subTest(option=option):
                    path = root / (option[2:] + ".pack")
                    path.write_bytes(b"before")
                    before = _evaluation_contract(evaluator, [option, str(path)])
                    path.write_bytes(b"after")
                    self.assertNotEqual(before, _evaluation_contract(evaluator, [option, str(path)]))
                    path.unlink()
                    with self.assertRaisesRegex(ValueError, "missing or not a file"):
                        _evaluation_contract(evaluator, [option, str(path)])
                    with self.assertRaisesRegex(ValueError, "missing its input path"):
                        _evaluation_contract(evaluator, [option])

    def test_world_pack_replacement_and_missing_force_revalidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator, world, policy, shader = (root / name for name in ("evaluator", "world", "policy", "shader"))
            trace, evidence, counter = root / "trace", root / "evidence", root / "counter"
            world.write_bytes(b"world-a")
            policy.write_bytes(b"policy")
            shader.write_bytes(b"shader")
            evaluator.write_text(
                "#!/bin/sh\n" +
                f"count=0; test -f '{counter}' && count=$(cat '{counter}'); count=$((count+1)); printf '%s' \"$count\" > '{counter}'\n" +
                f"printf '# step nq=1 scene_bodies=0 scene_stride=13 timestep=0.02 environment=0\\n1\\t0\\n' > '{trace}'\n" +
                "printf '%s\\n' '{}'\n", encoding="utf-8")
            evaluator.chmod(0o755)
            args = ["--world-pack", str(world), "--policy-pack", str(policy),
                    "--metallib", str(shader), "--state-trace", str(trace), "--steps", "1"]
            _evaluate(evaluator, args, evidence)
            _evaluate(evaluator, args, evidence)
            self.assertEqual(counter.read_text(), "1")
            world.write_bytes(b"world-b")
            _evaluate(evaluator, args, evidence)
            self.assertEqual(counter.read_text(), "2")
            world.unlink()
            with self.assertRaisesRegex(ValueError, "missing or not a file"):
                _evaluate(evaluator, args, evidence)
            self.assertEqual(counter.read_text(), "2")

    def test_old_contract_is_invalidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "evaluator"
            evaluator.write_bytes(b"evaluator")
            arguments = ["--task", "velocity"]
            old = {"evaluator": str(evaluator), "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(), "arguments": arguments, "file_fingerprints": {}}
            legacy = hashlib.sha256(json.dumps(old, sort_keys=True).encode("utf-8")).hexdigest()
            self.assertNotEqual(legacy, _evaluation_contract(evaluator, arguments))

    def test_multifile_inputs_bypass_cache_and_missing_dependency_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator, policy, shader = root / "evaluator", root / "policy", root / "shader"
            urdf, mesh, trace, evidence, counter = root / "robot.urdf", root / "mesh.stl", root / "trace", root / "evidence", root / "counter"
            for path in (policy, shader, urdf, mesh): path.write_text("fixture", encoding="utf-8")
            evaluator.write_text(
                "#!/bin/sh\n" + f"test -f '{mesh}' || exit 3\n" +
                f"count=0; test -f '{counter}' && count=$(cat '{counter}'); count=$((count+1)); printf '%s' \"$count\" > '{counter}'\n" +
                f"printf '# step nq=1 scene_bodies=0 scene_stride=13 timestep=0.02 environment=0\\n1\\t0\\n' > '{trace}'\n" +
                "printf '%s\\n' '{}'\n", encoding="utf-8")
            evaluator.chmod(0o755)
            args = ["--urdf", str(urdf), "--policy-pack", str(policy), "--metallib", str(shader), "--state-trace", str(trace), "--steps", "1"]
            self.assertFalse(_cache_reusable(args))
            _evaluate(evaluator, args, evidence)
            _evaluate(evaluator, args, evidence)
            self.assertEqual(counter.read_text(), "2")
            mesh.unlink()
            with self.assertRaises(subprocess.CalledProcessError): _evaluate(evaluator, args, evidence)
            self.assertEqual(counter.read_text(), "2")
            for option in ("--visual-observation-config", "--dove-manifest", "--numifly-wing-manifest", "--g1-visual-pack-dir", "--ball-visual-pack-dir"):
                self.assertFalse(_cache_reusable([option, "path"]))
            self.assertFalse(_cache_reusable(["--task", "numifly-flight"]))
            with self.assertRaisesRegex(ValueError, "missing or not a directory"):
                _evaluation_contract(evaluator, ["--g1-visual-pack-dir", str(root / "missing")])

    def test_evaluation_rejects_policy_mutated_during_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.pack"
            policy.write_bytes(b"before")
            shader = root / "shader.metallib"
            shader.write_bytes(b"shader")
            evaluator = root / "evaluator.sh"
            evaluator.write_text(
                "#!/bin/sh\n" + f"printf changed > '{policy}'\n" +
                "printf '%s\\n' '{}'\n", encoding="utf-8")
            evaluator.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                _evaluate(evaluator, ["--policy-pack", str(policy), "--metallib", str(shader),
                                      "--state-trace", str(root / "trace")], root / "evidence.json")
            self.assertFalse((root / "evidence.json").exists())

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


if __name__ == "__main__":
    unittest.main()
