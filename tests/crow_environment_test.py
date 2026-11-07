import copy
import unittest

from metalrobo.crow_environment import validate_first_episode


class FirstEpisodeAdmissionTest(unittest.TestCase):
    def setUp(self):
        # A first-attempt collision and an independently completed lane.
        self.evidence = {
            "steps_by_environment": [12, 40],
            "maximum_waypoints_by_environment": [2, 5],
            "completion_seen_by_environment": [False, True],
            "safe_completion_by_environment": [False, True],
            "physical_failure_by_environment": [True, False],
            "closed_by_environment": [True, True],
            "termination_reason_by_environment": [3, 7],
            "first_completion_step_by_environment": [None, 40],
            "safe_completion_count": 1, "completion_seen_count": 1,
            "physical_failure_count": 1, "closed_episode_count": 2,
            "horizon_censored_count": 0,
        }

    def test_valid_closed_and_censored_episodes(self):
        validate_first_episode(self.evidence, 2, 100)
        value = copy.deepcopy(self.evidence)
        value["steps_by_environment"][0] = 100
        value["physical_failure_by_environment"][0] = False
        value["closed_by_environment"][0] = False
        value["termination_reason_by_environment"][0] = 0
        value.update(physical_failure_count=0, closed_episode_count=1, horizon_censored_count=1)
        validate_first_episode(value, 2, 100)

    def test_rejects_forged_or_truncated_evidence(self):
        cases = [
            ("safe_completion_count", 2),
            ("physical_failure_by_environment", [False, False]),
            ("maximum_waypoints_by_environment", [2, float("nan")]),
            ("first_completion_step_by_environment", [None, 41]),
            ("steps_by_environment", [12]),
            ("closed_by_environment", [False, True]),
            ("safe_completion_by_environment", [True, True]),
        ]
        for field, replacement in cases:
            with self.subTest(field=field):
                value = copy.deepcopy(self.evidence)
                value[field] = replacement
                with self.assertRaises(RuntimeError):
                    validate_first_episode(value, 2, 100)


if __name__ == "__main__":
    unittest.main()
