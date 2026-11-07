import copy
import unittest

from metalrobo.crow_environment import validate_first_episode, validate_feeding_accounting


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


class FeedingAccountingAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.value = {"mean_reward": 4.25 / 1600,
            "outcomes": {"accepted_intake_utility": {"mean": 4.25 / 1600}}, "finite_resource_accounting": {
            "remaining_micrograms_by_environment": [0, 5000],
            "crop_micrograms_by_environment": [45000, 40000],
            "consumed_micrograms_by_environment": [45000, 40000],
            "bite_count_by_environment": [3, 2], "initialized_by_environment": [1, 1],
            "accepted_control_ticks_by_environment": [1601, 1601]}}

    def test_partial_bite_and_per_lane_completion(self):
        result = validate_feeding_accounting(self.value, 2, 1600)
        self.assertEqual(result["completed_lanes"], 1)
        self.assertEqual(result["mean_consumed_micrograms"], 42500)

    def test_rejects_refill_reward_and_clock_forgery(self):
        for field, replacement in (("remaining_micrograms", [45000, 5000]),
                                   ("crop_micrograms", [45000, 45000]),
                                   ("consumed_micrograms", [45001, 40000]),
                                   ("bite_count", [4, 2]),
                                   ("accepted_control_ticks", [1601, 1]),
                                   ("initialized", [True, 1])):
            with self.subTest(field=field):
                value = copy.deepcopy(self.value)
                value["finite_resource_accounting"][field + "_by_environment"] = replacement
                with self.assertRaises(RuntimeError):
                    validate_feeding_accounting(value, 2, 1600)
        self.value["mean_reward"] = 4.5 / 1600
        with self.assertRaises(RuntimeError):
            validate_feeding_accounting(self.value, 2, 1600)
        self.value["mean_reward"] = float("nan")
        with self.assertRaises(RuntimeError):
            validate_feeding_accounting(self.value, 2, 1600)

    def test_missing_lane_is_not_zero_food(self):
        self.value["finite_resource_accounting"]["remaining_micrograms_by_environment"] = [0]
        with self.assertRaises(RuntimeError):
            validate_feeding_accounting(self.value, 2, 1600)


if __name__ == "__main__":
    unittest.main()
