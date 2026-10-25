import importlib.util
import math
import pathlib
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("self_separation_176", HERE / "self_separation_176.py")
OWNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWNER)


def disjoint_pair(gap, offset=(0.0, 0.0, 0.0)):
    p = np.array([
        [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [gap, 0.0, 0.0], [gap, 1.0, 0.0], [gap, 0.0, 1.0],
    ], dtype=np.float64)
    p += np.asarray(offset, dtype=np.float64)
    return p, np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)


class PairMarginTests(unittest.TestCase):
    def test_repeated_threat_fractions_are_capped_only_for_disjoint_midplane(self):
        p, f = disjoint_pair(1.7e-6)
        for requested in (0.25, 0.55, 0.70, 0.75):
            with self.subTest(requested=requested):
                rows, records = OWNER.build_pair_rows(p, f, [(0, 1)], margin_fraction=requested)
                self.assertEqual(len(records), 1)
                record = records[0]
                effective = min(requested, OWNER.MAX_DISJOINT_MARGIN_FRACTION)
                self.assertAlmostEqual(record["requested_margin_fraction"], requested)
                self.assertAlmostEqual(record["effective_margin_fraction"], effective)
                self.assertEqual(record["margin_cap_applied"], requested > effective)
                self.assertTrue(all(float(row["lower"]) <= 2e-8 for row in rows))
                self.assertAlmostEqual(record["preserved_margin_mm"], effective * record["support_gap_mm"], delta=1e-18)

    def test_reported_007_violation_is_fixed_by_fraction_cap(self):
        reported_violation = 4.263383146962953e-7
        requested_fraction = 0.55
        inferred_gap = reported_violation / (requested_fraction - 0.5)
        p, f = disjoint_pair(inferred_gap)
        rows, records = OWNER.build_pair_rows(p, f, [(0, 1)], margin_fraction=requested_fraction)
        self.assertAlmostEqual((requested_fraction - 0.5) * records[0]["support_gap_mm"], reported_violation, delta=1e-18)
        self.assertEqual(records[0]["effective_margin_fraction"], 0.49)
        self.assertTrue(all(float(row["lower"]) <= 2e-8 for row in rows))

    def test_sub_half_fraction_keeps_existing_margin(self):
        p, f = disjoint_pair(0.25)
        rows, records = OWNER.build_pair_rows(p, f, [(0, 1)], margin_fraction=0.25)
        record = records[0]
        self.assertFalse(record["margin_cap_applied"])
        self.assertAlmostEqual(record["effective_margin_fraction"], 0.25)
        self.assertTrue(all(float(row["lower"]) <= 0.0 for row in rows))

    def test_common_vertex_wedge_retains_requested_fraction(self):
        p = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
        ], dtype=np.float64)
        f = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
        rows, records = OWNER.build_pair_rows(p, f, [(0, 1)], margin_fraction=0.75)
        self.assertEqual(records[0]["kind"], "common_vertex_wedge")
        self.assertEqual(records[0]["requested_margin_fraction"], 0.75)
        self.assertEqual(records[0]["effective_margin_fraction"], 0.75)
        self.assertFalse(records[0]["margin_cap_applied"])
        self.assertTrue(all(float(row["lower"]) <= 2e-8 for row in rows))

    def test_rows_are_translation_invariant_for_small_separations(self):
        p0, f = disjoint_pair(0.125)
        p1, _ = disjoint_pair(0.125, (1.0e6, -2.0e6, 3.0e6))
        rows0, rec0 = OWNER.build_pair_rows(p0, f, [(0, 1)], margin_fraction=0.75)
        rows1, rec1 = OWNER.build_pair_rows(p1, f, [(0, 1)], margin_fraction=0.75)
        self.assertEqual(rec0[0]["effective_margin_fraction"], rec1[0]["effective_margin_fraction"])
        self.assertAlmostEqual(rec0[0]["support_gap_mm"], rec1[0]["support_gap_mm"], delta=2e-9)
        self.assertEqual(len(rows0), len(rows1))
        for a, b in zip(rows0, rows1):
            np.testing.assert_array_equal(a["nodes"], b["nodes"])
            np.testing.assert_allclose(a["coefficients"], b["coefficients"], rtol=0.0, atol=2e-14)
            self.assertAlmostEqual(a["lower"], b["lower"], delta=2e-9)
            sums = [math.fsum(float(c[d]) for c in a["coefficients"]) for d in range(3)]
            self.assertLess(max(abs(x) for x in sums), 1e-14)

    def test_projection_relaxation_remains_below_capped_barrier(self):
        p, f = disjoint_pair(1.7e-6)
        rows, records = OWNER.build_pair_rows(p, f, [(0, 1)], margin_fraction=0.75)
        adjusted = OWNER.relax_pair_rows_for_projection(rows, records, 0.25 * records[0]["preserved_margin_mm"])
        self.assertAlmostEqual(adjusted[0]["effective_preserved_margin_mm"], 0.75 * records[0]["preserved_margin_mm"])
        self.assertTrue(all(np.isfinite(r["lower"]) for r in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
