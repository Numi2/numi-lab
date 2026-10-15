#!/usr/bin/env python3
"""Focused CPU contract tests for the bounded CGLS recurrence/layout."""
import unittest
import numpy as np
from scipy import sparse
import cgls_reference as cgls


class CGLSReferenceTests(unittest.TestCase):
    def setUp(self):
        # Deliberately nonsymmetric, diagonally dominant, full-rank 6x6 matrix.
        self.A = np.array([
            [4.0, 1.0, 0.0, 0.2, 0.0, 0.0],
            [0.1, 3.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 2.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 4.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.1, 3.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.2, 2.0],
        ], dtype=np.float32)
        self.rhs = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.5], dtype=np.float32)
        self.free_nodes = np.array([0, 2], dtype=np.uint32)

    def run_case(self, **kwargs):
        return cgls.run_cgls_fp32(sparse.csr_matrix(self.A), self.rhs,
                                  self.free_nodes, node_count=3, **kwargs)

    def test_nonsymmetric_recurrence_is_finite_and_descending(self):
        result = self.run_case(cap=32)
        self.assertIn(result.status, (cgls.EXHAUSTED, cgls.ZERO_NORMAL_RESIDUAL))
        self.assertGreater(result.iterations, 0)
        self.assertTrue(np.isfinite(result.direction).all())
        self.assertTrue(np.isfinite(result.residual).all())
        free_solution = np.concatenate([result.direction[0, :3],
                                        result.direction[2, :3]])
        action = self.A @ free_solution
        start_norm = float(np.linalg.norm(self.rhs.astype(np.float64)))
        final_norm = float(np.linalg.norm(self.rhs.astype(np.float64) - action.astype(np.float64)))
        self.assertLess(final_norm, start_norm)
        self.assertGreater(float(np.dot(self.rhs.astype(np.float64), action.astype(np.float64))), 0.0)
        self.assertEqual(np.count_nonzero(result.direction[1, :3]), 0)
        self.assertEqual(np.count_nonzero(result.direction[3:, :]), 0)

    def test_bad_caps_fail_closed(self):
        for cap in (0, 1, 16, 31, 33, 65, -1):
            with self.subTest(cap=cap):
                result = self.run_case(cap=cap)
                self.assertEqual(result.status, cgls.INVALID_INPUT)
                self.assertEqual(result.iterations, 0)

    def test_nonzero_fixed_rhs_fails_closed(self):
        bad_rhs, mask, row_env = cgls.make_layout(self.rhs, self.free_nodes, 3)
        bad_rhs[1, 0] = 0.75  # node 1, which is fixed
        result = cgls.run_cgls_fp32(sparse.csr_matrix(self.A), bad_rhs,
            self.free_nodes, node_count=3, row_mask=mask, environment_map=row_env)
        self.assertEqual(result.status, cgls.INVALID_INPUT)
        self.assertIn("fixed", result.error or "")

    def test_quiet_environment_must_be_exact_zero(self):
        result = self.run_case(selected=(1, 0), cap=32)
        self.assertNotEqual(result.status, cgls.INVALID_INPUT)
        # The helper's full layout has env1 after the first N mechanics rows.
        _, mask, row_env = cgls.make_layout(self.rhs, self.free_nodes, 3)
        mask[3, 0] = 1.0
        result = cgls.run_cgls_fp32(sparse.csr_matrix(self.A), self.rhs,
            self.free_nodes, node_count=3, row_mask=mask, environment_map=row_env)
        self.assertEqual(result.status, cgls.INVALID_INPUT)
        self.assertIn("quiet", result.error or "")

    def test_nonbinary_mask_and_bad_environment_map_fail_closed(self):
        rhs, mask, row_env = cgls.make_layout(self.rhs, self.free_nodes, 3)
        mask[0, 1] = 0.5
        result = cgls.run_cgls_fp32(sparse.csr_matrix(self.A), rhs,
            self.free_nodes, node_count=3, row_mask=mask, environment_map=row_env)
        self.assertEqual(result.status, cgls.INVALID_INPUT)

        _, mask, row_env = cgls.make_layout(self.rhs, self.free_nodes, 3)
        row_env[0] = 1
        result = cgls.run_cgls_fp32(sparse.csr_matrix(self.A), rhs,
            self.free_nodes, node_count=3, row_mask=mask, environment_map=row_env)
        self.assertEqual(result.status, cgls.INVALID_INPUT)
        self.assertIn("environment", result.error or "")

    def test_small_matrix_has_an_explicit_transpose(self):
        A = sparse.csr_matrix(self.A)
        x = np.array([0.2, -0.7, 0.3, 1.1, -0.4, 0.6], dtype=np.float64)
        y = np.array([1.0, 0.25, -0.5, 0.3, 2.0, -1.0], dtype=np.float64)
        self.assertGreater(np.linalg.norm((A - A.T).toarray()), 0.1)
        lhs = float(y @ (A @ x))
        rhs = float((A.T @ y) @ x)
        self.assertAlmostEqual(lhs, rhs, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
