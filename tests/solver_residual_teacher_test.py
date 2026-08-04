import unittest

from metalrobo.solver_residual_teacher import select_physical_frontier


class SolverResidualTeacherTest(unittest.TestCase):
    def test_frontier_keeps_progress_stability_tradeoffs(self) -> None:
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
        candidates, selected = select_physical_frontier(
            evidence, environment_count=4, elite_count=4
        )
        self.assertTrue(candidates[1].pareto)  # greatest peak progress
        self.assertTrue(candidates[2].pareto)  # stable best final progress
        self.assertFalse(candidates[3].pareto)  # dominated everywhere
        self.assertEqual(set(selected), {1, 2})
        self.assertEqual(selected[0], 2)

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
            select_physical_frontier(
                evidence, environment_count=1, elite_count=1
            )


if __name__ == "__main__":
    unittest.main()
