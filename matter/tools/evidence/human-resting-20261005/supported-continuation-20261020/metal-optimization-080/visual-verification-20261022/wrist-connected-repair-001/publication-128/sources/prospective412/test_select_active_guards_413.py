import unittest
import numpy as np

from select_active_guards_413 import select_threatened_pairs


class ActiveGuardSelectionTests(unittest.TestCase):
    def payload(self):
        # Pair (10,11) has two rows and one is threatened; (20,21) is
        # prospective but safe; (30,31) is mandatory and safe.
        return dict(
            row_indptr=np.array([0, 1, 2, 3, 4], dtype=np.int64),
            row_node_ids=np.array([0, 1, 2, 3], dtype=np.int32),
            row_coefficients=np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]),
            row_lower_mm=np.array([-0.2, -1.0, -1.0, -0.1]),
            row_pair_ids=np.array([[10, 11], [10, 11], [20, 21], [30, 31]], dtype=np.int32),
        )

    def test_selects_whole_threatened_pair_and_keeps_mandatory(self):
        d = np.zeros((4, 3)); d[0, 0] = -0.5
        selected, deferred, rows, slack, report = select_threatened_pairs(
            **self.payload(), predicted_delta_mm=d,
            mandatory_pair_ids=np.array([[30, 31]], dtype=np.int32),
        )
        np.testing.assert_array_equal(selected, [[10, 11], [30, 31]])
        np.testing.assert_array_equal(deferred, [[20, 21]])
        np.testing.assert_array_equal(rows, [0, 1, 3])
        self.assertEqual(report["threatened_pair_count"], 1)
        self.assertEqual(report["selected_pair_count"], 2)
        self.assertTrue(report["zero_step_feasible_for_selected_rows"])

    def test_node_trust_clips_prediction_and_reports_it(self):
        d = np.zeros((4, 3)); d[0, 0] = -2.0
        selected, _, _, _, report = select_threatened_pairs(
            **self.payload(), predicted_delta_mm=d,
            mandatory_pair_ids=np.array([[30, 31]], dtype=np.int32),
            max_node_step_mm=1.0,
        )
        np.testing.assert_array_equal(selected, [[10, 11], [30, 31]])
        self.assertEqual(report["predicted_clipped_node_count"], 1)
        self.assertAlmostEqual(report["predicted_max_node_step_after_clip_mm"], 1.0)

    def test_zero_activity_margin_does_not_select_safe_pairs(self):
        d = np.zeros((4, 3))
        selected, deferred, _, _, report = select_threatened_pairs(
            **self.payload(), predicted_delta_mm=d,
            mandatory_pair_ids=np.array([[30, 31]], dtype=np.int32),
        )
        np.testing.assert_array_equal(selected, [[30, 31]])
        np.testing.assert_array_equal(deferred, [[10, 11], [20, 21]])
        self.assertEqual(report["threatened_row_count"], 0)

    def test_missing_mandatory_pair_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "mandatory pair IDs are missing"):
            select_threatened_pairs(
                **self.payload(), predicted_delta_mm=np.zeros((4, 3)),
                mandatory_pair_ids=np.array([[40, 41]], dtype=np.int32),
            )

    def test_malformed_nonfinite_or_overspecified_payload_fails_closed(self):
        p = self.payload(); d = np.zeros((4, 3)); d[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            select_threatened_pairs(**p, predicted_delta_mm=d, mandatory_pair_ids=np.array([[30, 31]]))
        with self.assertRaisesRegex(ValueError, "canonical"):
            select_threatened_pairs(**p, predicted_delta_mm=np.zeros((4, 3)), mandatory_pair_ids=np.array([[11, 10]]))

    def test_activity_margin_selects_near_active_pair(self):
        d = np.zeros((4, 3))
        selected, deferred, _, slack, report = select_threatened_pairs(
            **self.payload(), predicted_delta_mm=d,
            mandatory_pair_ids=np.array([[30, 31]], dtype=np.int32),
            activity_margin_mm=0.25,
        )
        np.testing.assert_array_equal(selected, [[10, 11], [30, 31]])
        np.testing.assert_array_equal(deferred, [[20, 21]])
        self.assertTrue(np.all(np.isfinite(slack)))
        self.assertEqual(report["threatened_pair_count"], 2)


if __name__ == "__main__":
    unittest.main()
