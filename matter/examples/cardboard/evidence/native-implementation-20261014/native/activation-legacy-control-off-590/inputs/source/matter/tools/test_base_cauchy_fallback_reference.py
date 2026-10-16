"""Independent CPU contracts for the bounded base-state Cauchy fallback.

These tests check the mathematical row-selection and least-squares direction
contract. They do not qualify the Metal implementation; the source133 native
kernel checker separately exercises the production kernels and captured world.
"""

import math
import unittest


def capture_rhs(rhs, fixed, inverse_mass, node_count, environment_count):
    """Reference the kernel's E*N mechanics mask and full field-half copy."""
    total_nodes = node_count * environment_count
    if len(rhs) != 2 * total_nodes or len(fixed) != total_nodes or len(inverse_mass) != total_nodes:
        raise ValueError("expected exactly the mechanical and field backing blocks")
    result = [(0.0, 0.0, 0.0, 0.0)] * len(rhs)
    for row in range(total_nodes):
        if fixed[row] == 0.0 and math.isfinite(inverse_mass[row]) and inverse_mass[row] > 0.0:
            x, y, z, _ = rhs[row]
            result[row] = (x, y, z, 0.0)
    result[total_nodes:] = rhs[total_nodes:]
    return result


def zero_finite_field_rows(rhs, node_count, environment_count):
    total_nodes = node_count * environment_count
    return all(math.isfinite(value) and value == 0.0
               for row in rhs[total_nodes:] for value in row)


def finite_free_mechanical_rows(rhs, free):
    if len(rhs) != len(free):
        raise ValueError("mechanical RHS and free-row mask lengths differ")
    return all(math.isfinite(component)
               for row, is_free in zip(rhs, free) if is_free
               for component in row[:3])


def raw_cauchy(matrix, rhs, free):
    """Return p=tau*A^T*b on the already-eliminated free subspace."""
    n = len(rhs)
    if len(matrix) != n or any(len(row) != n for row in matrix) or len(free) != n:
        raise ValueError("square operator and matching free-row mask required")
    b = [rhs[i] if free[i] else 0.0 for i in range(n)]
    gradient = [math.fsum(matrix[i][j] * b[i] for i in range(n))
                if free[j] else 0.0 for j in range(n)]
    action = [math.fsum(matrix[i][j] * gradient[j] for j in range(n))
              if free[i] else 0.0 for i in range(n)]
    gradient_norm2 = math.fsum(value * value for value in gradient)
    action_norm2 = math.fsum(value * value for value in action)
    if not math.isfinite(gradient_norm2) or not math.isfinite(action_norm2) or \
            gradient_norm2 <= 0.0 or action_norm2 <= 0.0:
        raise ValueError("Cauchy fallback requires finite nonzero gradient and action")
    tau = gradient_norm2 / action_norm2
    return [tau * value for value in gradient], gradient, action, tau


class BaseCauchyReferenceTests(unittest.TestCase):
    def test_full_rhs_preserves_field_rows_and_masks_fixed_mechanics(self):
        rhs = [
            (2.0, -1.0, 4.0, 9.0),
            (8.0, 3.0, -2.0, 7.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ]
        captured = capture_rhs(rhs, [0.0, 1.0], [1.0, 0.0], 2, 1)
        self.assertEqual(captured[0], (2.0, -1.0, 4.0, 0.0))
        self.assertEqual(captured[1], (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(captured[2:], rhs[2:])
        self.assertTrue(zero_finite_field_rows(captured, 2, 1))

    def test_any_nonzero_or_nonfinite_field_rhs_fails_closed(self):
        for value in (1.0e-40, float("inf"), float("nan")):
            rhs = [(0.0, 0.0, 0.0, 0.0)] * 4
            rhs[2] = (0.0, value, 0.0, 0.0)
            captured = capture_rhs(rhs, [0.0, 0.0], [1.0, 1.0], 2, 1)
            self.assertFalse(zero_finite_field_rows(captured, 2, 1))

    def test_nonfinite_free_base_residual_is_not_sanitized_into_a_direction(self):
        for value in (float("inf"), float("nan")):
            rhs = [(value, 0.0, 0.0, 0.0), (4.0, 1.0, -2.0, 0.0)]
            self.assertFalse(finite_free_mechanical_rows(rhs, [True, False]))
        self.assertTrue(finite_free_mechanical_rows(
            [(float("nan"), 9.0, 3.0, 0.0), (4.0, 1.0, -2.0, 0.0)],
            [False, True]))

    def test_raw_cauchy_is_descent_and_zeroes_fixed_rows(self):
        matrix = [[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 5.0]]
        rhs = [1.0, 2.0, 99.0]
        direction, gradient, action, tau = raw_cauchy(
            matrix, rhs, [True, True, False])
        self.assertEqual(direction[2], 0.0)
        self.assertAlmostEqual(tau, 65.0 / 850.0, places=15)
        base2 = math.fsum(value * value for value in rhs[:2])
        predicted2 = math.fsum(
            (rhs[i] - tau * action[i]) ** 2 for i in range(2))
        self.assertLess(predicted2, base2)
        self.assertLess(-math.fsum(gradient[i] ** 2 for i in range(2)), 0.0)

    def test_zero_or_singular_action_is_not_admitted(self):
        with self.assertRaises(ValueError):
            raw_cauchy([[0.0, 0.0], [0.0, 0.0]], [1.0, 0.0], [True, True])


if __name__ == "__main__":
    unittest.main()
