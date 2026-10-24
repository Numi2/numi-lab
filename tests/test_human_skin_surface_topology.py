import unittest
import numpy as np
from human_skin_patch_geometry import (surface_topology_signature as topology,
    require_matching_surface_topology as match)


def torus(n=7, m=9):
    faces = []
    index = lambda i, j: (i % n)*m + (j % m)
    for i in range(n):
        for j in range(m):
            a, b = index(i, j), index(i+1, j)
            c, d = index(i+1, j+1), index(i, j+1)
            faces.extend(((a, b, c), (a, c, d)))
    return np.asarray(faces, dtype=np.int64)


class TopologyTests(unittest.TestCase):
    sphere = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])

    def test_disk_uses_only_referenced_rows(self):
        # Sparse identities do not imply 100,000 mechanical coordinates.
        p = topology(np.array([[1, 99, 99999], [1, 99999, 17]]))[0]
        self.assertEqual((p['used_vertex_count'], p['edge_count'],
                          p['euler_characteristic'], p['genus']), (4, 5, 1, 0))
        self.assertEqual(len(p['boundary_loops']), 1)

    def test_closed_sphere(self):
        p = topology(self.sphere)[0]
        self.assertEqual((p['euler_characteristic'], p['genus']), (2, 0))
        self.assertEqual(p['boundary_loops'], [])

    def test_detects_handle_with_one_boundary(self):
        p = topology(torus()[1:])[0]
        self.assertEqual((p['euler_characteristic'], p['genus']), (-1, 1))
        self.assertEqual(len(p['boundary_loops']), 1)
        with self.assertRaises(ValueError):
            match(self.sphere[1:], torus()[1:])

    def test_closed_large_torus(self):
        p = topology(torus(37, 31))[0]
        self.assertEqual((p['euler_characteristic'], p['genus']), (0, 1))

    def test_annulus(self):
        f = np.array([[i, (i+1) % 4, 4+(i+1) % 4] for i in range(4)] +
                     [[i, 4+(i+1) % 4, 4+i] for i in range(4)])
        p = topology(f)[0]
        self.assertEqual((len(p['boundary_loops']), p['genus']), (2, 0))

    def test_disconnected_components(self):
        result = topology(np.vstack((self.sphere, self.sphere+10)))
        self.assertEqual(len(result), 2)
        self.assertEqual([p['genus'] for p in result], [0, 0])
        with self.assertRaises(ValueError):
            match(self.sphere, np.vstack((self.sphere, self.sphere+10)))

    def test_vertex_pinched_spheres(self):
        second = self.sphere+10
        second[second == 10] = 0
        with self.assertRaisesRegex(ValueError, 'disconnected vertex link'):
            topology(np.vstack((self.sphere, second)))

    def test_invalid_faces(self):
        for f in (np.array([[0, 0, 1]]), np.array([[0, 1, 2], [0, 1, 2]]),
                  np.array([[0, 1, 2], [0, 1, 3]]), np.array([[-1, 2, 3]]),
                  np.array([[0., 1., 2.]]), np.array([[0, 1]])):
            with self.subTest(f=f), self.assertRaises(ValueError):
                topology(f)

    def test_triangulation_and_ids_may_change(self):
        a = np.array([[0, 1, 2], [0, 2, 3]])
        b = np.array([[12, 13, 14], [12, 14, 15], [12, 15, 16]])
        self.assertEqual(match(a, b), [(1, 0)])

    def test_empty(self):
        self.assertEqual(topology(np.empty((0, 3), np.int64)), [])
        with self.assertRaises(ValueError):
            match(np.empty((0, 3), np.int64), self.sphere)


if __name__ == '__main__':
    unittest.main()
