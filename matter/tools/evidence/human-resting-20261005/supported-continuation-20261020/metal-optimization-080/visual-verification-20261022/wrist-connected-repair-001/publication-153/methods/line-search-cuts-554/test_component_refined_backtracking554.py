import unittest
import numpy as np

from component_refined_backtracking554 import component_line_search


def run(base, target, valid, refine_steps=6):
    accepted = {"points": base.copy()}
    calls = []

    def gate(points):
        alpha = float(points[0, 0] * 1000.0)
        calls.append(alpha)
        passed = bool(valid(alpha))
        if passed:
            accepted["points"] = points.copy()
        return passed, {"observed_alpha": alpha, "edge_gate_pass": passed}

    out, records = component_line_search(
        base, target, np.array([0]), [np.array([0])], gate, refine_steps=refine_steps
    )
    # Mirrors 549 main's final gate call: the incremental cache is asked to
    # certify the exact returned Float32 pose after search completion.
    final_ok, _ = gate(out)
    return out, records[0], calls, accepted["points"], final_ok


class ComponentRefinedBacktracking554Tests(unittest.TestCase):
    def test_near_one_edge_gate_refines_after_power_two_pass(self):
        base = np.zeros((1, 3), dtype=np.float32)
        target = np.array([[.001, 0, 0]], dtype=np.float32)
        out, record, calls, cached, final_ok = run(
            base, target, lambda a: a <= .9, refine_steps=6
        )
        refinement = record["alpha_refinement_554"]
        self.assertEqual(record["selected_alpha"], .8984375)
        self.assertTrue(refinement["attempted"])
        self.assertEqual(refinement["initial_bracket"], {
            "passing_alpha": .5, "failed_alpha": 1.0, "bracket_width": .5
        })
        self.assertEqual(len(refinement["trials"]), 6)
        self.assertEqual(calls[:3], [0.0, 1.0, .5])
        self.assertGreater(record["selected_alpha"], .5)
        self.assertLessEqual(record["selected_alpha"], .9)
        self.assertTrue(final_ok)
        self.assertEqual(out.tobytes(), cached.tobytes())

    def test_nonmonotonic_gate_records_actual_failures_and_keeps_latest_pass(self):
        base = np.zeros((1, 3), dtype=np.float32)
        target = np.array([[.001, 0, 0]], dtype=np.float32)
        # A disconnected passing island at .70 is intentionally not sampled;
        # the search makes no monotonicity/global-max claim.
        out, record, calls, cached, final_ok = run(
            base, target, lambda a: (abs(a) < 1e-9 or .49 < a < .51 or
                                     .62 < a < .63 or .699 < a < .701),
            refine_steps=4,
        )
        refinement = record["alpha_refinement_554"]
        self.assertEqual(record["selected_alpha"], .625)
        self.assertEqual([t["pass"] for t in refinement["trials"]],
                         [False, True, False, False])
        self.assertEqual([t["alpha"] for t in refinement["trials"]],
                         [.75, .625, .6875, .65625])
        self.assertNotIn(.70, [t["alpha"] for t in refinement["trials"]])
        # .70 is outside the bounded probe schedule; no search-wide maximum
        # or monotonicity statement is made.
        self.assertEqual(refinement["search_claim"],
                         "bounded_actual_gate_probes_only_no_monotonicity_or_global_maximum_claim")
        self.assertTrue(final_ok)
        self.assertEqual(out.tobytes(), cached.tobytes())

    def test_no_pass_returns_none_and_does_not_refine(self):
        base = np.zeros((1, 3), dtype=np.float32)
        target = np.array([[.001, 0, 0]], dtype=np.float32)
        out, record, _calls, cached, final_ok = run(
            base, target, lambda a: a == 0.0, refine_steps=6
        )
        self.assertIsNone(record["selected_alpha"])
        self.assertFalse(record["alpha_refinement_554"]["attempted"])
        self.assertTrue(np.array_equal(out, base))
        self.assertTrue(final_ok)
        self.assertEqual(out.tobytes(), cached.tobytes())

    def test_refinement_steps_are_actual_gate_calls_and_cache_ends_at_output(self):
        base = np.zeros((1, 3), dtype=np.float32)
        target = np.array([[.001, 0, 0]], dtype=np.float32)
        out, record, calls, cached, final_ok = run(
            base, target, lambda a: a <= .8, refine_steps=3
        )
        self.assertEqual(calls, [0.0, 1.0, .5, .75, .875, .8125, float(out[0, 0] * 1000.0)])
        self.assertEqual(len(record["alpha_refinement_554"]["trials"]), 3)
        self.assertTrue(final_ok)
        self.assertEqual(out.tobytes(), cached.tobytes())


if __name__ == "__main__":
    unittest.main()

