"""Regression checks for connected local skin patch boundary construction."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "human_skin_patch_geometry",
    Path(__file__).resolve().parents[1] / "tools/human_skin_patch_geometry.py")
geom = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geom)


class AnnularPatchTests(unittest.TestCase):
    def setUp(self):
        a = np.arange(25)*2*np.pi/25
        b = np.arange(154)*2*np.pi/154 + .37
        self.x = np.r_[np.c_[np.cos(a), np.sin(a), a*0],
                       np.c_[.98*np.cos(b), .98*np.sin(b), b*0+.05]]
        self.a = np.arange(25)
        self.b = np.arange(25, 179)

    def test_unequal_loops_form_oriented_annulus(self):
        faces, report = geom.zipper_annulus(
            self.x, self.a, self.b, max_edge=.32)
        self.assertEqual(len(faces), 179)
        self.assertTrue(report["oriented_boundary_exact"])
        self.assertFalse(report["physical_admission"])
        edges = {}
        for face in faces:
            for i, j in zip(face, np.roll(face, -1)):
                edges.setdefault(tuple(sorted((int(i), int(j)))), []).append((int(i), int(j)))
        self.assertEqual(len(self.x)-len(edges)+len(faces), 0)
        actual = {v[0] for v in edges.values() if len(v) == 1}
        expected = set(zip(self.a, np.roll(self.a, -1)))
        expected |= set(zip(np.roll(self.b, -1), self.b))
        self.assertEqual(actual, expected)
        self.assertGreater(report["minimum_quality"], .18)
        self.assertLessEqual(report["maximum_edge_m"], .32)

    def test_cyclic_reindexing_has_same_optimum(self):
        f, r = geom.zipper_annulus(self.x, self.a, self.b, max_edge=.32)
        g, s = geom.zipper_annulus(self.x, self.a, np.roll(self.b, 17), max_edge=.32)
        self.assertAlmostEqual(r["score"], s["score"], places=12)
        canonical = lambda faces: {min(tuple(np.roll(t, k)) for k in range(3)) for t in faces}
        self.assertEqual(canonical(f), canonical(g))

    def test_impossible_edge_cap_rejected(self):
        with self.assertRaisesRegex(ValueError, "no strip"):
            geom.zipper_annulus(self.x, self.a, self.b, max_edge=.001)

    def test_invalid_indices_rejected(self):
        for a, b in [([0, 1, 1], [25, 26, 27]),
                     ([0, 1, 2], [2, 3, 4]),
                     ([0., 1., 2.], [25, 26, 27]),
                     ([-1, 1, 2], [25, 26, 27]),
                     ([0, 1, 999], [25, 26, 27])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                geom.zipper_annulus(self.x, a, b)

    def test_invalid_geometry_and_limits_rejected(self):
        bad = self.x.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            geom.zipper_annulus(bad, self.a, self.b)
        for limit in [-1, 0, np.inf, np.nan]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                geom.zipper_annulus(self.x, self.a, self.b, max_edge=limit)

    def test_collinear_surface_rejected(self):
        x = np.c_[np.arange(6), np.zeros((6, 2))]
        with self.assertRaisesRegex(ValueError, "no strip"):
            geom.zipper_annulus(x, [0, 1, 2], [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
