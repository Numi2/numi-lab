#!/usr/bin/env python3
"""Focused CPU tests for regional source-face attribution."""
import unittest

from select_skin_fat_refinement_faces import select_faces_by_coverage, regional_failure


def face(donor, strain, nodes):
    return {"donor": donor, "strain": strain, "force_nodes": set(nodes)}


class RegionalRefinementSelectionTests(unittest.TestCase):
    def test_multi_region_strain_and_force_are_covered_jointly(self):
        faces = {
            (0, 1, 2): face(99, 4.0, {0, 1, 2}),
            (3, 4, 5): face(20, 6.0, {3, 4, 5}),
            (6, 7, 8): face(99, 6.0, {6, 7, 8}),
        }
        force_error = {0: 0.2, 1: 0.1, 3: 0.4, 4: 0.3, 6: 0.15, 7: 0.05}
        node_region = {0: 99, 1: 99, 2: 99, 3: 20, 4: 20, 5: 20,
                       6: 99, 7: 99, 8: 99}
        result = select_faces_by_coverage(
            faces, force_error, node_region,
            {99: 10.0, 20: 6.0}, {99: 0.5, 20: 0.7},
            0.8, 0.8, {99}, {20}, 8)
        self.assertGreaterEqual(result["selected_region_strain"][99], 8.0)
        self.assertGreaterEqual(result["selected_region_force"][20], 0.56)
        self.assertGreaterEqual(result["strain_fraction"], 0.8)
        self.assertGreaterEqual(result["force_fraction"], 0.8)
        self.assertIn((3, 4, 5), result["selected"])
        self.assertTrue(any(faces[f]["donor"] == 99 for f in result["selected"]))

    def test_face_limit_fails_closed_if_region_coverage_is_incomplete(self):
        faces = {
            (0, 1, 2): face(99, 1.0, {0, 1, 2}),
            (3, 4, 5): face(20, 9.0, {3, 4, 5}),
        }
        with self.assertRaisesRegex(ValueError, "face limit"):
            select_faces_by_coverage(
                faces, {}, {}, {99: 1.0, 20: 9.0}, {},
                0.99, 0.99, {99}, {20}, 1)

    def test_zero_reference_with_nonzero_force_is_failed_region(self):
        failed, relative = regional_failure(1.0, 0.0, 0.01)
        self.assertTrue(failed)
        self.assertIsNone(relative)
        self.assertEqual(regional_failure(0.0, 0.0, 0.01), (False, 0.0))


if __name__ == "__main__":
    unittest.main()
