#!/usr/bin/env python3
"""Analytical checks for the aortic cavity Schur scaling geometry."""

import math
import unittest


FACES = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
REST = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def mul(s, a):
    return tuple(s * x for x in a)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def volume(points, origin):
    total = 0.0
    for i, j, k in FACES:
        a, b, c = (sub(points[n], origin) for n in (i, j, k))
        total += dot(a, cross(b, c)) / 6.0
    return total


def volume_gradient(points, fixed_origin):
    gradient = [(0.0, 0.0, 0.0) for _ in points]
    for i, j, k in FACES:
        a, b, c = (sub(points[n], fixed_origin) for n in (i, j, k))
        gradient[i] = add(gradient[i], mul(1.0 / 6.0, cross(b, c)))
        gradient[j] = add(gradient[j], mul(1.0 / 6.0, cross(c, a)))
        gradient[k] = add(gradient[k], mul(1.0 / 6.0, cross(a, b)))
    return gradient


def simpson_force_gradient(start, end, fixed_origin):
    midpoint = tuple(mul(0.5, add(a, b)) for a, b in zip(start, end))
    gradient = [(0.0, 0.0, 0.0) for _ in start]
    for i, j, k in FACES:
        terms = []
        for points in (start, midpoint, end):
            a, b, c = (sub(points[n], fixed_origin) for n in (i, j, k))
            terms.append((mul(1.0 / 6.0, cross(b, c)),
                          mul(1.0 / 6.0, cross(c, a)),
                          mul(1.0 / 6.0, cross(a, b))))
        for vertex, node in enumerate((i, j, k)):
            averaged = mul(1.0 / 6.0, add(add(terms[0][vertex],
                mul(4.0, terms[1][vertex])), terms[2][vertex]))
            gradient[node] = add(gradient[node], averaged)
    return gradient


def face_lumped_schur(start, end, inverse_diagonal, dt,
                      pressure_scale, equation_scale):
    fixed_origin = start[FACES[0][0]]
    total = 0.0
    midpoint = tuple(mul(0.5, add(a, b)) for a, b in zip(start, end))
    for i, j, k in FACES:
        grads = []
        for points in (start, midpoint, end):
            a, b, c = (sub(points[n], fixed_origin) for n in (i, j, k))
            grads.append((mul(1.0 / 6.0, cross(b, c)),
                          mul(1.0 / 6.0, cross(c, a)),
                          mul(1.0 / 6.0, cross(a, b))))
        a1, b1, c1 = (sub(end[n], fixed_origin) for n in (i, j, k))
        volume_gradients = (mul(1.0 / 6.0, cross(b1, c1)),
                            mul(1.0 / 6.0, cross(c1, a1)),
                            mul(1.0 / 6.0, cross(a1, b1)))
        for vertex, node in enumerate((i, j, k)):
            g_volume = volume_gradients[vertex]
            g_force = mul(1.0 / 6.0, add(add(grads[0][vertex],
                mul(4.0, grads[1][vertex])), grads[2][vertex]))
            contribution = dot(g_volume, tuple(d * g for d, g in
                zip(inverse_diagonal[node], g_force)))
            total += max(contribution, 0.0)
    return dt * dt * (pressure_scale / equation_scale) * total


class CavitySchurReferenceTests(unittest.TestCase):
    def setUp(self):
        self.start = REST
        # Small affine-like deformation, while retaining the same fixed origin
        # used by the production cavity volume and its discrete pressure work.
        self.end = ((0.0002, -0.0001, 0.0003),
                    (1.04, 0.02, -0.01),
                    (0.03, 0.97, 0.015),
                    (-0.02, 0.01, 1.03))
        self.origin = self.start[FACES[0][0]]

    def test_candidate_volume_gradient_is_exact(self):
        gradient = volume_gradient(self.end, self.origin)
        epsilon = 1.0e-6
        for node in range(len(self.end)):
            for axis in range(3):
                plus = list(self.end)
                minus = list(self.end)
                delta = tuple(epsilon if i == axis else 0.0 for i in range(3))
                plus[node] = add(plus[node], delta)
                minus[node] = sub(minus[node], delta)
                numerical = (volume(plus, self.origin) -
                             volume(minus, self.origin)) / (2.0 * epsilon)
                self.assertAlmostEqual(numerical, gradient[node][axis], places=8)

    def test_simpson_force_gradient_preserves_exact_cubic_volume_work(self):
        gradient = simpson_force_gradient(self.start, self.end, self.origin)
        work = sum(dot(g, sub(x1, x0)) for g, x0, x1 in
                   zip(gradient, self.start, self.end))
        difference = (volume(self.end, self.origin) -
                      volume(self.start, self.origin))
        self.assertAlmostEqual(work, difference, places=12)

    def test_face_lumped_schur_estimate_is_positive_and_scaled(self):
        inverse = ((0.0, 0.0, 0.0), (0.2, 0.25, 0.3),
                   (0.3, 0.2, 0.25), (0.25, 0.3, 0.2))
        estimate = face_lumped_schur(
            self.start, self.end, inverse, 1.0 / 240.0, 13300.0, 0.001)
        unit_scaled = face_lumped_schur(
            self.start, self.end, inverse, 1.0 / 240.0, 1.0, 1.0)
        doubled_dt = face_lumped_schur(
            self.start, self.end, inverse, 2.0 / 240.0, 13300.0, 0.001)
        self.assertTrue(math.isfinite(estimate))
        self.assertGreater(estimate, 1.0e-12)
        self.assertAlmostEqual(estimate, unit_scaled * 13300.0 / 0.001,
                               delta=abs(estimate) * 1.0e-12)
        self.assertAlmostEqual(doubled_dt, 4.0 * estimate,
                               delta=abs(estimate) * 1.0e-12)


if __name__ == "__main__":
    unittest.main()
