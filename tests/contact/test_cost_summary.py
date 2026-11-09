#!/usr/bin/env python3
import unittest
from cost_summary import summarize


class CostSummaryTest(unittest.TestCase):
    def sample(self, be, mid):
        out = []
        for i, (c, m) in enumerate(zip(be, mid), 1):
            for arm, rate in (("ipc_area", c), ("ipc_area_midpoint", m)):
                out.append({"arm": arm, "repetition": i, "accepted_and_matches_accuracy_state": True,
                            "accepted_seconds_per_wall_second": rate})
        if len(be) >= 2:
            out[2:4] = reversed(out[2:4])
        return out

    def test_three_pairs_and_ten_percent_are_both_required(self):
        yes = summarize(self.sample([1, 1, 1], [1.12, 1.11, 1.13]))
        self.assertTrue(yes["complete"] and yes["practical_benefit"])
        self.assertFalse(summarize(self.sample([1, 1, 1], [1.2, 1.2, .9]))["practical_benefit"])
        self.assertFalse(summarize(self.sample([1, 1, 1], [1.09, 1.09, 1.09]))["practical_benefit"])
        self.assertFalse(summarize(self.sample([1, 1], [1.2, 1.2]))["complete"])

    def test_failed_and_duplicate_runs_are_not_successes(self):
        runs = self.sample([1, 1, 1], [1.2, 1.2, 1.2])
        runs[0]["accepted_and_matches_accuracy_state"] = False
        self.assertFalse(summarize(runs)["complete"])
        runs[0]["accepted_and_matches_accuracy_state"] = True
        with self.assertRaises(ValueError):
            summarize(runs + [runs[0]])


if __name__ == "__main__":
    unittest.main()
