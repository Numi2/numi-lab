import unittest

from metalrobo.solver_residual_teacher import rank_physical_realizations


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


if __name__ == "__main__":
    unittest.main()
