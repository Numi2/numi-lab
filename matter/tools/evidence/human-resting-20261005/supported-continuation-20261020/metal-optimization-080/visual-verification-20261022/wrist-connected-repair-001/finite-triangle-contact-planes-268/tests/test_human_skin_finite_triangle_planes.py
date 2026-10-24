import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1] / 'tools/human_skin_contact_planes.py'
if not path.exists():
    path = Path('/tmp/human_skin_contact_planes_268.py')
spec = importlib.util.spec_from_file_location('human_skin_contact_planes', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
planes = module.finite_triangle_support_planes


class FiniteTrianglePlaneTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]) * .001

    def test_coplanar_disjoint_requires_in_plane_axis(self):
        b = self.a + [.003, 0., 0.]
        result = planes(self.a, b)
        self.assertGreater(result[0]['gap_mm'], 0.)
        self.assertIn('in_plane', result[0]['axis'])
        self.assertEqual(result, sorted(result, key=lambda row: -row['gap_mm']))

    def test_crossing_support_constraints_cover_all_corners(self):
        b = np.array([[.5,-.5,-1],[.5,1.5,1],[.5,1.5,-1]]) * .001
        rows = planes(self.a, b)
        self.assertLessEqual(rows[0]['gap_mm'], 0.)
        for row in rows:
            n = row['normal']
            pairs = (self.a @ n)[:,None] - (b @ n)[None,:]
            self.assertAlmostEqual(row['gap_mm'], pairs.min()*1000, places=12)
            margin = .2
            moved = self.a + n * ((margin-row['gap_mm'])/1000)
            gaps = ((moved @ n)[:,None] - (b @ n)[None,:])*1000
            self.assertGreaterEqual(gaps.min(), margin-1e-12)
            # A selected previous intersection-point row cannot prove this
            # property: all nine finite corner combinations are required.
            self.assertAlmostEqual(gaps.min(), margin, places=12)

    def test_preferred_solid_outward_hemisphere(self):
        b = self.a + [0.,0.,.0005]
        preferred = np.array([0.,0.,1.])
        rows = planes(self.a, b, preferred_normal=preferred)
        self.assertTrue(rows)
        self.assertTrue(all(row['normal'] @ preferred > 0 for row in rows))
        self.assertAlmostEqual(rows[0]['gap_mm'], -.5)
        self.assertAlmostEqual(np.linalg.norm(rows[0]['normal']), 1.)

    def test_permutation_and_rigid_transform_invariance(self):
        b = np.array([[.5,-.5,-1],[.5,1.5,1],[.5,1.5,-1]]) * .001
        theta = .713
        rotation = np.array([[np.cos(theta),-np.sin(theta),0],
                             [np.sin(theta),np.cos(theta),0],[0,0,1]])
        before = planes(self.a, b)[0]['gap_mm']
        after = planes(self.a[[2,0,1]] @ rotation.T + .017,
                       b[[1,0,2]] @ rotation.T + .017)[0]['gap_mm']
        self.assertAlmostEqual(before, after, places=11)
        moved_origin = planes(self.a+1e6, b+1e6)[0]['gap_mm']
        self.assertAlmostEqual(before, moved_origin, places=6)

    def test_tiny_valid_triangle_is_not_rejected_by_absolute_tolerance(self):
        result = planes(self.a*1e-20, (self.a+[.003,0,0])*1e-20)
        self.assertTrue(result)
        self.assertGreater(result[0]['gap_mm'], 0.)
        self.assertTrue(all(np.isfinite(row['gap_mm']) for row in result))

    def test_invalid_input_rejected(self):
        for bad in (np.zeros((3,3)), self.a[:2], self.a*np.nan):
            with self.assertRaises(ValueError): planes(bad, self.a)
        for bad in ([0,0,0],[1,np.inf,0],[1,0]):
            with self.assertRaises(ValueError): planes(self.a, self.a, bad)

    def test_random_support_bounds_and_reversal(self):
        rng = np.random.default_rng(268)
        for _ in range(100):
            a, b = rng.normal(size=(2,3,3))*.001
            rows = planes(a,b)
            reverse = planes(b,a)
            self.assertAlmostEqual(rows[0]['gap_mm'], reverse[0]['gap_mm'], places=11)
            for row in rows:
                n = row['normal']
                ia, ib = row['a_support'], row['b_support']
                gaps = ((a @ n)[:,None] - (b @ n)[None,:])*1000
                self.assertAlmostEqual(gaps.min(), row['gap_mm'], places=11)
                self.assertAlmostEqual(gaps[ia,ib], gaps.min(), places=11)

if __name__ == '__main__': unittest.main()
