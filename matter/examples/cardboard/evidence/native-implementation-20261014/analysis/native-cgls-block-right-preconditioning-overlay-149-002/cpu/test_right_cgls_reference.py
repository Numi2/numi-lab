#!/usr/bin/env python3
import unittest
import numpy as np
import right_cgls_reference as ref

class RightCGLSTests(unittest.TestCase):
    def setUp(self):
        self.A = np.array([
            [5., .7, -.2, .15, 0., .1, 0., 0., 0.],
            [-.3, 4.2, .8, 0., -.1, 0., .05, 0., 0.],
            [.1, -.4, 3.8, 0., .08, .12, 0., .03, 0.],
            [.2, 0., 0., 4.8, .3, -.5, 0., .1, 0.],
            [.05, .1, 0., -.6, 3.9, .4, .12, 0., .05],
            [0., -.1, .2, .5, -.2, 4.4, 0., .1, .2],
            [0., 0., .04, 0., .1, 0., 4.1, -.3, .2],
            [.1, 0., 0., 0., 0., .1, .5, 3.7, .4],
            [0., .02, 0., .1, -.1, 0., -.2, .3, 4.6],
        ], dtype=np.float32)
        self.b = np.array([1., -2., .3, .7, 2.2, -1.1, -.2, .8, 1.3], dtype=np.float32)
        b0 = np.array([[2.0, .3, -.1], [-.2, 1.5, .25], [.05, -.15, 1.2]], np.float32)
        b1 = np.array([[1.7, -.2, .1], [.4, 2.2, -.3], [-.15, .2, 1.4]], np.float32)
        b2 = np.array([[2.4, .15, .35], [-.1, 1.8, .2], [.3, -.25, 1.6]], np.float32)
        self.S = np.zeros((9, 9), np.float32)
        for i, block in enumerate((b0, b1, b2)):
            self.S[3*i:3*i+3, 3*i:3*i+3] = np.linalg.inv(block).astype(np.float32)

    def test_right_recurrence_reduces_original_unscaled_residual(self):
        out = ref.run_cgls_fp32(self.A, self.b, cap=64, scale_blocks=self.S)
        self.assertIn(out.status, (ref.EXHAUSTED, ref.ZERO_NORMAL))
        self.assertGreater(out.iterations, 0)
        self.assertTrue(np.isfinite(out.x).all())
        self.assertLess(np.linalg.norm(out.residual.astype(np.float64)),
                        np.linalg.norm(self.b.astype(np.float64)))
        self.assertGreater(float(self.b.astype(np.float64) @
                                 (self.A.astype(np.float64) @ out.x.astype(np.float64))), 0.0)

    def test_transpose_is_required_for_nonsymmetric_scale(self):
        self.assertGreater(float(np.linalg.norm(self.S.astype(np.float64) -
                                                self.S.T.astype(np.float64))), 1e-3)
        correct = ref.run_cgls_fp32(self.A, self.b, cap=32,
                                    scale_blocks=self.S, use_transpose=True)
        wrong = ref.run_cgls_fp32(self.A, self.b, cap=32,
                                  scale_blocks=self.S, use_transpose=False)
        self.assertTrue(np.isfinite(correct.x).all())
        self.assertTrue(np.isfinite(wrong.x).all())
        self.assertGreater(float(np.linalg.norm(correct.x.astype(np.float64) -
                                                wrong.x.astype(np.float64))), 1e-5)

    def test_default_unscaled_recurrence_is_same_as_identity_transform(self):
        omitted = ref.run_cgls_fp32(self.A, self.b, cap=32)
        explicit = ref.run_cgls_fp32(self.A, self.b, cap=32,
                                     scale_blocks=np.eye(9, dtype=np.float32))
        np.testing.assert_array_equal(omitted.x, explicit.x)
        np.testing.assert_array_equal(omitted.residual, explicit.residual)

    def test_singular_node_block_is_not_regularized_or_accepted(self):
        singular = np.array([[1., 2., 0.], [2., 4., 0.], [0., 0., 1.]], np.float64)
        with self.assertRaises(np.linalg.LinAlgError):
            np.linalg.inv(singular)
        # Admission validates a computed inverse; nonfinite or absent rows fail.
        tags = np.array([[0., 1., 0.], [0., 0., 0.]], np.float32)
        masses = np.ones_like(tags)
        mask = np.zeros((2, 3, 4), np.float32)
        mask[0, 0, :3] = 1
        mask[0, 2, :3] = 1
        blocks = np.zeros((2, 3, 3, 4), np.float32)
        blocks[0, 0, :3, :3] = self.S[:3, :3]
        blocks[0, 0, 0, 3] = 1
        blocks[0, 2, :3, :3] = self.S[6:9, 6:9]
        blocks[0, 2, 0, 3] = 1
        selected = np.array([1, 0], np.uint32)
        blocks[0, 2, 1, 3] = np.nan
        ok, reason = ref.validate_tags_and_blocks(tags, masses, mask, blocks, selected)
        self.assertFalse(ok)
        self.assertEqual(reason, "invalid-block")

    def test_human_tag_and_wrong_fixed_quiet_masks_fail_closed(self):
        tags = np.array([[0., 1., 2.], [0., 0., 0.]], np.float32)
        masses = np.ones_like(tags)
        mask = np.zeros((2, 3, 4), np.float32)
        mask[0, 0, :3] = 1
        blocks = np.zeros((2, 3, 3, 4), np.float32)
        blocks[0, 0, :3, :3] = self.S[:3, :3]
        blocks[0, 0, 0, 3] = 1
        ok, reason = ref.validate_tags_and_blocks(
            tags, masses, mask, blocks, np.array([1, 0], np.uint32))
        self.assertFalse(ok)
        self.assertEqual(reason, "unsupported-constraint")
        tags[0, 2] = 1
        mask[0, 2, 0] = 1
        ok, reason = ref.validate_tags_and_blocks(
            tags, masses, mask, blocks, np.array([1, 0], np.uint32))
        self.assertFalse(ok)
        self.assertEqual(reason, "fixed-row")

    def test_field_and_quiet_rows_are_exactly_zero(self):
        E, N = 2, 3
        U = 2 * E * N
        vector = np.zeros((U, 4), np.float32)
        mask = np.zeros_like(vector)
        row_env = np.array([0,0,0,1,1,1,0,0,0,1,1,1], np.uint32)
        selected = np.array([1,0], np.uint32)
        blocks = np.zeros((E,N,3,4), np.float32)
        for node in (0,2):
            blocks[0,node,:3,:3] = self.S[3*node:3*node+3,3*node:3*node+3]
            blocks[0,node,0,3] = 1
            mask[node,:3] = 1
        vector[0,:3] = [1.,-2.,.5]
        vector[2,:3] = [.25,1.5,-.75]
        scaled = ref.apply_right_scale(vector, mask, row_env, selected, blocks, N)
        self.assertTrue(np.all(scaled[3:] == 0))
        bad = vector.copy(); bad[6,0] = 1.0  # a field backing row
        with self.assertRaisesRegex(ValueError, "quiet/fixed/field"):
            ref.apply_right_scale(bad, mask, row_env, selected, blocks, N)

if __name__ == "__main__":
    unittest.main(verbosity=2)
