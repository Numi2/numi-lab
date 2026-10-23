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


class LayerPrismTests(unittest.TestCase):
    def test_all_permutations_match_native_sorted_id_signs(self):
        import itertools
        faces = np.asarray(list(itertools.permutations((0, 1, 2))), dtype=np.int64)
        nodes, layers, signs = geom.ordered_layer_prism_tets(faces)
        np.testing.assert_array_equal(nodes[:, 0], nodes[:, 1])
        np.testing.assert_array_equal(layers[:, 1], layers[:, 0] + 1)
        expected = np.array([-1, 1, 1, -1, -1, 1])
        np.testing.assert_array_equal(signs[:, 0, 0], expected)
        np.testing.assert_array_equal(signs[:, 0], signs[:, 1])
        # Compare against a direct native-template decomposition, not a second
        # inversion-count implementation.
        o = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        native_nodes = np.array([[0, 1, 2, 2], [0, 1, 1, 2], [0, 0, 1, 2]])
        for i, face in enumerate(faces):
            inward = np.cross(o[face[1]]-o[face[0]], o[face[2]]-o[face[0]])
            xyz = np.stack((o, o-.3*inward, o-.9*inward))
            p = xyz[layers[i], nodes[i]]
            det, _ = geom.tetra_determinants_and_gradients(p)
            self.assertTrue(np.all(signs[i]*det > 0))
            np.testing.assert_array_equal(nodes[i, 0], native_nodes)
            self.assertAlmostEqual(float(np.abs(det[0]).sum()/6), .15)
            self.assertAlmostEqual(float(np.abs(det[1]).sum()/6), .30)

    def test_empty_and_invalid_faces(self):
        n, l, s = geom.ordered_layer_prism_tets(np.empty((0, 3), dtype=np.int64))
        self.assertEqual(n.shape, (0, 2, 3, 4))
        self.assertEqual(l.shape, n.shape)
        self.assertEqual(s.shape, (0, 2, 3))
        for f in [[[0, 1, 1]], [[-1, 2, 3]], [[0., 1., 2.]], [[0, 1]]]:
            with self.subTest(f=f), self.assertRaises(ValueError):
                geom.ordered_layer_prism_tets(f)

    def test_batch_signed_and_absolute_volume_gradients(self):
        rng = np.random.default_rng(20261023)
        p = rng.normal(size=(24, 4, 3))*.07
        d = rng.normal(size=p.shape)
        d /= np.linalg.norm(d, axis=(1, 2))[:, None, None]
        det, gradient = geom.tetra_determinants_and_gradients(p)
        h = 2e-7
        plus, _ = geom.tetra_determinants_and_gradients(p+h*d)
        minus, _ = geom.tetra_determinants_and_gradients(p-h*d)
        np.testing.assert_allclose((plus-minus)/(2*h), np.sum(gradient*d, axis=(1, 2)),
                                   rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose((np.abs(plus)-np.abs(minus))/(12*h),
                                   np.sign(det)*np.sum(gradient*d, axis=(1, 2))/6,
                                   rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose(gradient.sum(axis=-2), 0., atol=1e-17)
        for invalid in [np.zeros((3, 3)), np.full((4, 3), np.nan)]:
            with self.assertRaises(ValueError):
                geom.tetra_determinants_and_gradients(invalid)

if __name__ == "__main__":
    unittest.main()
