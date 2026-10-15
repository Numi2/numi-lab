#!/usr/bin/env python3
import math
import unittest

from mortar_union_partition import (
    PartitionError,
    Patch,
    covered_patch_ids_at,
    partition_side,
    paired_contact_is_suppressed,
)


E0 = (1.0, 0.0, 0.0)
E1 = (0.0, 1.0, 0.0)
E2 = (0.0, 0.0, 1.0)
MID12 = (0.0, 0.5, 0.5)
CENTROID = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def patch(stable_id, triangle, *, side_b=None, active=True):
    return Patch(stable_id, tuple(triangle), tuple(side_b or triangle), active)


class MortarUnionPartitionTests(unittest.TestCase):
    def assert_area(self, partition, covered, uncovered):
        self.assertAlmostEqual(partition.covered_area, covered, places=11)
        self.assertAlmostEqual(partition.uncovered_area, uncovered, places=11)
        self.assertAlmostEqual(partition.area_error, 0.0, places=10)
        self.assertTrue(all(
            (cell.vertices[1][0] - cell.vertices[0][0]) *
            (cell.vertices[2][1] - cell.vertices[0][1]) -
            (cell.vertices[1][1] - cell.vertices[0][1]) *
            (cell.vertices[2][0] - cell.vertices[0][0]) > 0.0
            for cell in partition.cells))

    def test_full_parent_patch_has_empty_complement(self):
        result = partition_side([patch(12, [E0, E1, E2])], "a")
        self.assert_area(result, 0.5, 0.0)
        self.assertEqual(len(result.cells), 1)

    def test_three_sector_union_conserves_area_and_identity(self):
        sectors = [
            patch(10, [E0, E1, CENTROID]),
            patch(20, [E1, E2, CENTROID]),
            patch(30, [E2, E0, CENTROID]),
        ]
        result = partition_side(sectors, "a")
        self.assert_area(result, 0.5, 0.0)
        self.assertEqual([cell.patch_id for cell in result.cells if cell.covered],
                         [10, 20, 30])

    def test_same_patch_identity_suppresses_paired_closure(self):
        sectors = [
            patch(10, [E0, E1, CENTROID]),
            patch(20, [E1, E2, CENTROID]),
            patch(30, [E2, E0, CENTROID]),
        ]
        self.assertTrue(paired_contact_is_suppressed(
            sectors, (0.6, 0.2, 0.2), (0.6, 0.2, 0.2)))
        # The center lies on all three closed sector boundaries. Both sides
        # share the same nonempty ID set, so deterministic edge ties remain
        # classified as bonded closure.
        self.assertTrue(paired_contact_is_suppressed(
            sectors, CENTROID, CENTROID))

    def test_different_covered_patch_ids_do_not_suppress_contact(self):
        sectors = [
            patch(10, [E0, E1, CENTROID]),
            patch(20, [E1, E2, CENTROID]),
            patch(30, [E2, E0, CENTROID]),
        ]
        # Identity-only side maps model a nonmatching cross-patch event: both
        # points lie in the union, but the paired face identity differs.
        self.assertFalse(paired_contact_is_suppressed(
            sectors, (0.45, 0.45, 0.1), (0.1, 0.45, 0.45)))

    def test_shared_edge_suppression_requires_common_identity(self):
        left_patch = patch(10, [E0, E1, MID12],
                           side_b=[E0, E1, MID12])
        right_patch = patch(20, [E0, MID12, E2],
                            side_b=[E0, MID12, E2])
        self.assertTrue(paired_contact_is_suppressed(
            [left_patch, right_patch], E0, E0))
        self.assertFalse(paired_contact_is_suppressed(
            [left_patch, right_patch], (1.0 / 3.0, 0.5, 1.0 / 6.0),
            (1.0 / 3.0, 1.0 / 6.0, 0.5)))

    def test_two_neighbors_share_edge_and_cover_exactly(self):
        neighbors = [
            patch(2, [E0, E1, MID12]),
            patch(3, [E0, MID12, E2]),
        ]
        result = partition_side(neighbors, "a")
        self.assert_area(result, 0.5, 0.0)
        edge_ids = covered_patch_ids_at(neighbors, "a", MID12)
        self.assertEqual(edge_ids, (2, 3))
        self.assertTrue(covered_patch_ids_at(neighbors, "a", E0))

    def test_partial_patch_preserves_uncovered_complement(self):
        partial = patch(7, [
            (0.6, 0.2, 0.2),
            (0.2, 0.6, 0.2),
            (0.2, 0.2, 0.6),
        ])
        result = partition_side([partial], "a")
        self.assert_area(result, 0.08, 0.42)
        self.assertTrue(covered_patch_ids_at([partial], "a", (0.4, 0.3, 0.3)))
        self.assertEqual(covered_patch_ids_at([partial], "a", (0.9, 0.05, 0.05)), ())

    def test_each_side_uses_its_own_map(self):
        side_a = patch(1, [E0, E1, MID12], side_b=[E0, E1, E2])
        a = partition_side([side_a], "a")
        b = partition_side([side_a], "b")
        self.assert_area(a, 0.25, 0.25)
        self.assert_area(b, 0.5, 0.0)

    def test_dormant_patch_does_not_enter_active_union(self):
        active = patch(1, [E0, E1, MID12])
        dormant_overlap = patch(2, [E0, E1, MID12], active=False)
        result = partition_side([active, dormant_overlap], "a")
        self.assert_area(result, 0.25, 0.25)
        self.assertEqual(covered_patch_ids_at(
            [active, dormant_overlap], "a", (0.2, 0.1, 0.7)), ())

    def test_positive_area_overlap_is_rejected(self):
        a = patch(1, [E0, E1, MID12])
        b = patch(2, [E0, E1, E2])
        with self.assertRaisesRegex(PartitionError, "overlap by area"):
            partition_side([a, b], "a")

    def test_overlap_is_checked_independently_on_each_parent_side(self):
        left = patch(1, [E0, E1, MID12], side_b=[E0, E1, E2])
        right = patch(2, [E0, MID12, E2], side_b=[E0, E1, E2])
        self.assert_area(partition_side([left, right], "a"), 0.5, 0.0)
        with self.assertRaisesRegex(PartitionError, "overlap by area"):
            partition_side([left, right], "b")

    def test_bounded_workspace_overflow_fails_closed(self):
        partial = patch(1, [
            (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)
        ])
        with self.assertRaisesRegex(PartitionError, "bounded workspace"):
            partition_side([partial], "a", max_triangles=1)
        with self.assertRaisesRegex(PartitionError, "bounded workspace"):
            partition_side([
                patch(1, [E0, E1, MID12]),
                patch(2, [E0, MID12, E2]),
            ], "a", max_triangles=1)

    def test_nonfinite_and_outside_inputs_reject(self):
        with self.assertRaisesRegex(PartitionError, "finite"):
            partition_side([patch(1, [E0, E1, (float("nan"), 0.0, 1.0)])], "a")
        with self.assertRaisesRegex(PartitionError, "outside"):
            partition_side([patch(1, [E0, E1, (1.2, -0.1, -0.1)])], "a")

    def test_duplicate_active_ids_reject(self):
        with self.assertRaisesRegex(PartitionError, "identifiers must be unique"):
            partition_side([
                patch(4, [E0, E1, MID12]),
                patch(4, [E0, MID12, E2]),
            ], "a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
