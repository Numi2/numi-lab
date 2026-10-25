import unittest
import numpy as np
from surface_displacement_prolongation import prolongate_surface_displacement


def grid(n=7):
    xyz = np.array([[float(i), float(j), 0.0] for j in range(n) for i in range(n)])
    f = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i; b = a + 1; c = a + n; d = c + 1
            f.extend(((a, b, d), (a, d, c)))
    return xyz, np.asarray(f, dtype=np.int64)


class ProlongationTests(unittest.TestCase):
    def test_direct_anchor_exact_and_boundary_fixed(self):
        v, f = grid()
        anchor = 3 * 7 + 3
        d = np.array([[0.0, 0.0, 0.2]])
        result = prolongate_surface_displacement(v, f, [anchor], d, [], np.empty((0, 3)), np.empty((0, 3)))
        out = result["displacements_m"]
        self.assertTrue(np.allclose(out[anchor], d[0], atol=1e-12))
        edge_counts = {}
        for tri in f:
            for a, b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
                key = tuple(sorted((int(a),int(b))))
                edge_counts[key] = edge_counts.get(key,0) + 1
        boundary = np.unique([x for edge, count in edge_counts.items() if count == 1 for x in edge])
        self.assertTrue(np.all(out[boundary] == 0))
        self.assertGreater(result["report"]["support_faces"], 0)

    def test_barycentric_anchor_exact(self):
        v, f = grid()
        face = 28
        weights = np.array([[0.2, 0.3, 0.5]])
        target = np.array([[0.0, 0.0, 0.1]])
        r = prolongate_surface_displacement(v, f, [], np.empty((0, 3)), [face], weights, target)
        got = weights[0] @ r["displacements_m"][f[face]]
        self.assertTrue(np.allclose(got, target[0], atol=2e-8))

    def test_conflicting_duplicate_direct_anchor_rejected(self):
        v, f = grid()
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            prolongate_surface_displacement(v, f, [24, 24], [[0,0,0.1],[0,0,0.2]], [], np.empty((0,3)), np.empty((0,3)))

    def test_invalid_barycentric_weights_rejected(self):
        v, f = grid()
        with self.assertRaisesRegex(ValueError, "invalid barycentric"):
            prolongate_surface_displacement(v, f, [], np.empty((0,3)), [0], [[0.2,0.2,0.2]], [[0,0,0.1]])

    def test_out_of_range_anchor_rejected(self):
        v, f = grid()
        with self.assertRaisesRegex(ValueError, "outside surface"):
            prolongate_surface_displacement(v, f, [len(v)], [[0,0,1]], [], np.empty((0,3)), np.empty((0,3)))


if __name__ == "__main__":
    unittest.main()
