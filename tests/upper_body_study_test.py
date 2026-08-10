import unittest

import numpy as np

from metalrobo.upper_body_study import _metrics


class UpperBodyStudyTest(unittest.TestCase):
    def test_qualified_success_requires_motion_and_stability(self):
        trace = np.zeros((4, 50), dtype=np.float64)
        trace[:, 1 + 7 + 22] = (0.35, 0.1, -0.2, -0.3)
        result = {
            "failed_environment_steps": 0,
            "termination_count": 0,
            "maximum_tilt": 0.3,
            "minimum_root_height_by_environment": [0.7],
            "mean_tracking_score": 0.8,
        }
        motion = np.zeros((4, 19), dtype=np.float64)
        motion[:, 12] = (0.68, 0.80, 0.91, 0.89)
        measured = _metrics(result, trace, motion, (22,), ("right",))
        self.assertTrue(measured["qualified_success"])
        self.assertAlmostEqual(measured["shoulders"][0]["excursion_rad"], 0.65)

    def test_raised_motion_with_a_termination_is_rejected(self):
        trace = np.zeros((2, 50), dtype=np.float64)
        trace[:, 1 + 7 + 15] = (0.35, -0.4)
        motion = np.zeros((2, 19), dtype=np.float64)
        motion[:, 3] = (0.68, 0.90)
        result = {
            "failed_environment_steps": 0,
            "termination_count": 1,
            "maximum_tilt": 0.51,
            "minimum_root_height_by_environment": [0.7],
            "mean_tracking_score": 0.8,
        }
        measured = _metrics(result, trace, motion, (15,), ("left",))
        self.assertTrue(measured["raised"])
        self.assertFalse(measured["qualified_success"])

    def test_already_raised_static_pose_is_not_motion_success(self):
        trace = np.zeros((2, 50), dtype=np.float64)
        trace[:, 1 + 7 + 15] = (-0.4, -0.41)
        motion = np.zeros((2, 19), dtype=np.float64)
        motion[:, 3] = (0.91, 0.92)
        result = {
            "failed_environment_steps": 0,
            "termination_count": 0,
            "maximum_tilt": 0.2,
            "minimum_root_height_by_environment": [0.7],
            "mean_tracking_score": 0.8,
        }
        self.assertFalse(
            _metrics(result, trace, motion, (15,), ("left",))["raised"]
        )

    def test_bilateral_success_requires_both_wrist_outcomes(self):
        trace = np.zeros((3, 50), dtype=np.float64)
        motion = np.zeros((3, 19), dtype=np.float64)
        motion[:, 3] = (0.68, 0.90, 0.91)
        motion[:, 12] = (0.68, 0.82, 0.87)
        result = {
            "failed_environment_steps": 0,
            "termination_count": 0,
            "maximum_tilt": 0.2,
            "minimum_root_height_by_environment": [0.7],
            "mean_tracking_score": 0.8,
        }
        measured = _metrics(
            result, trace, motion, (15, 22), ("left", "right")
        )
        self.assertFalse(measured["raised"])


if __name__ == "__main__":
    unittest.main()
