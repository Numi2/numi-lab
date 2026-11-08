import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from metalrobo.solver_residual_teacher import (
    _soft_advantage_weights,
    distill_residual_frontier,
    rank_physical_realizations,
)


class SolverResidualTeacherTest(unittest.TestCase):
    def test_every_realization_contributes_in_continuous_rank_order(self) -> None:
        evidence = {
            "peak_forward_progress_by_environment_m": [1.0, 1.4, 1.2, 0.5],
            "final_forward_progress_by_environment_m": [0.9, 1.0, 1.15, 0.2],
            "mean_tracking_score_by_environment": [0.5, 0.55, 0.6, 0.1],
            "mean_root_height_by_environment": [0.6, 0.55, 0.65, 0.2],
            "minimum_root_height_by_environment": [0.4, 0.3, 0.5, 0.1],
            "mean_tilt_by_environment": [0.3, 0.5, 0.2, 1.0],
            "maximum_tilt_by_environment": [0.6, 0.9, 0.4, 1.5],
            "termination_count_by_environment": [0, 0, 0, 1],
            "physics_error_count_by_environment": [0, 0, 0, 0],
        }
        candidates, ranked = rank_physical_realizations(
            evidence,
            environment_count=4,
            mean_rewards=[0.4, 0.5, 0.7, -1.0],
        )
        self.assertEqual(set(ranked), {0, 1, 2, 3})
        self.assertEqual(ranked[0], 2)
        self.assertEqual(ranked[-1], 3)
        self.assertTrue(candidates[2].pareto)
        self.assertFalse(candidates[3].pareto)

    def test_world_tilt_and_height_do_not_rank_generic_motion(self) -> None:
        evidence = {
            "peak_forward_progress_by_environment_m": [0.0, 0.0],
            "final_forward_progress_by_environment_m": [0.0, 0.0],
            "mean_tracking_score_by_environment": [0.9, 0.4],
            "mean_root_height_by_environment": [0.1, 0.8],
            "minimum_root_height_by_environment": [0.0, 0.7],
            "mean_tilt_by_environment": [2.5, 0.0],
            "maximum_tilt_by_environment": [3.1, 0.1],
            "termination_count_by_environment": [0, 0],
            "physics_error_count_by_environment": [0, 0],
        }
        candidates, ranked = rank_physical_realizations(
            evidence,
            environment_count=2,
            mean_rewards=[0.8, 0.2],
        )
        self.assertEqual(ranked, [0, 1])
        self.assertGreater(candidates[0].quality, candidates[1].quality)

    def test_missing_per_environment_evidence_is_rejected(self) -> None:
        evidence = {
            key: [0.0]
            for key in (
                "peak_forward_progress_by_environment_m",
                "final_forward_progress_by_environment_m",
                "mean_tracking_score_by_environment",
                "mean_root_height_by_environment",
                "minimum_root_height_by_environment",
                "mean_tilt_by_environment",
                "maximum_tilt_by_environment",
                "termination_count_by_environment",
            )
        }
        with self.assertRaisesRegex(ValueError, "physics_error_count"):
            rank_physical_realizations(
                evidence, environment_count=1
            )

    def test_soft_advantage_weighting_emphasizes_without_exclusion(self) -> None:
        weights = _soft_advantage_weights(
            np.asarray([0.0, 0.5, 1.0]),
            temperature=6.0,
        )
        self.assertTrue(np.all(weights > 0.0))
        self.assertAlmostEqual(float(weights[-1]), 1.0)
        self.assertGreater(float(weights[-1] / weights[0]), 400.0)

    def test_soft_advantage_temperature_rejects_negative_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature"):
            _soft_advantage_weights(
                np.asarray([0.0, 1.0]),
                temperature=-1.0,
            )

    def test_distillation_rejects_physics_failure_before_ranking_or_training(self) -> None:
        try:
            import metalrobo.mlx_policy_learning  # noqa: F401
        except ImportError:
            self.skipTest("MLX is not available")
        dtype = [("reward", "f4"), ("done", "u4"), ("physics_error", "u4")]
        transitions = np.zeros((2, 2), dtype=dtype)
        transitions["reward"] = [[0.2, 0.4], [0.6, 0.8]]
        rollout = SimpleNamespace(
            transitions=transitions, control_step_count=2,
            environment_count=2, sample_count=4,
        )
        evidence = {
            "peak_forward_progress_by_environment_m": [0.0, 0.0],
            "final_forward_progress_by_environment_m": [0.0, 0.0],
            "mean_tracking_score_by_environment": [0.4, 0.5],
            "mean_root_height_by_environment": [0.6, 0.6],
            "minimum_root_height_by_environment": [0.5, 0.5],
            "mean_tilt_by_environment": [0.0, 0.0],
            "maximum_tilt_by_environment": [0.0, 0.0],
            "termination_count_by_environment": [0, 0],
            "physics_error_count_by_environment": [0, 0],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_path = root / "evidence.json"
            evidence_path.write_text(json.dumps(evidence))
            args = dict(
                policy_pack_path=root / "source.policy",
                rollout_pack_path=root / "source.rollout",
                evidence_path=evidence_path,
                output_policy_path=root / "result.policy",
                output_deployment_policy_path=root / "deploy.policy",
                output_evidence_path=root / "result.json",
                learning_rate=0.001, update_epochs=1, minibatch_size=4,
                environment_advantage_temperature=1.0,
                step_advantage_temperature=1.0, seed=1,
            )
            with (
                patch(
                    "metalrobo.mlx_policy_learning.read_policy_rollout_pack",
                    return_value=rollout,
                ),
                patch(
                    "metalrobo.mlx_policy_learning.MLXPolicyLearner.from_policy_pack",
                    side_effect=RuntimeError("learner reached"),
                ) as learner,
            ):
                # A failure in either environment at any step must not reach the learner.
                for index in ((0, 0), (1, 1)):
                    transitions["physics_error"].fill(0)
                    transitions["physics_error"][index] = 1
                    with self.subTest(index=index):
                        with self.assertRaisesRegex(ValueError, "physics failure"):
                            distill_residual_frontier(**args)
                        learner.assert_not_called()
                # Contradictory aggregate evidence must not supply ranks either.
                transitions["physics_error"].fill(0)
                evidence["physics_error_count_by_environment"] = [0, 1]
                evidence_path.write_text(json.dumps(evidence))
                with self.assertRaisesRegex(ValueError, "evidence contains a physics failure"):
                    distill_residual_frontier(**args)
                learner.assert_not_called()
                # An all-valid rollout must still reach the normal learner path.
                evidence["physics_error_count_by_environment"] = [0, 0]
                evidence_path.write_text(json.dumps(evidence))
                with self.assertRaisesRegex(RuntimeError, "learner reached"):
                    distill_residual_frontier(**args)
                learner.assert_called_once()
            self.assertFalse((root / "result.policy").exists())
            self.assertFalse((root / "deploy.policy").exists())
            self.assertFalse((root / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
