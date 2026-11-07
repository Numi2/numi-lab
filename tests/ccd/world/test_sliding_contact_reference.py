#!/usr/bin/env python3
"""Analytic sliding-impact counterexample for coupled contact updates."""
import unittest


def iterate_sliding(normal_first, iterations):
    # W=[[1,.4],[.4,2]]; incoming normal=-2 and tangent=100.
    # mu=.5; this reference has one unit-mass generalized constraint.
    n = t = 0.0
    for _ in range(iterations):
        rn = 2 - n - .4*t
        rt = -100 - .4*n - 2*t
        if normal_first:
            dn = max(n + rn, 0) - n
            pt = t + (rt - .4*dn)/2
            pn = n + dn
        else:
            pn = max(n + (2*rn - .4*rt)/1.84, 0)
            pt = t + (-.4*rn + rt)/1.84
        pt = max(-.5*pn, min(.5*pn, pt))
        n, t = pn, pt
    return n, t, -2 + n + .4*t, 100 + .4*n + 2*t


class SlidingReferenceTests(unittest.TestCase):
    def test_sliding_normal_balance_has_closed_form_solution(self):
        # Coulomb sliding gives t=-mu*n, hence 0=-2+(1-.4*.5)*n.
        # Large positive outgoing slip verifies the sliding assumption.
        n, t, vn, vt = iterate_sliding(True, 32)
        self.assertAlmostEqual(n, 2.5, places=10)
        self.assertAlmostEqual(t, -1.25, places=10)
        self.assertAlmostEqual(vn, 0, places=10)
        self.assertGreater(vt, 90)

    def test_projecting_full_inverse_has_wrong_fixed_point(self):
        n, t, vn, vt = iterate_sliding(False, 256)
        self.assertGreater(n, 0)
        self.assertGreater(vt, 0)
        self.assertAlmostEqual(t, -.5*n)
        self.assertGreater(abs(vn), 1.0)


if __name__ == "__main__":
    unittest.main()
