#!/usr/bin/env python3
"""Bounded synthetic tests for the source-derived selector-5 CPU reference."""
from __future__ import annotations
import math
import unittest

import regional_rap_reference as rap
import selector5_graph_reference as g


def fixture(*, graph_valid=True, rap_valid=True, tail_latch=False, augmentation=False):
    # Two overlapping patches: {0,1} and {1,2}. Node 2 is constrained. The
    # shared node exercises coverage averaging and the CSR's membership-index
    # convention. Local A blocks are nonsymmetric; patch 0 requires a row pivot.
    a0 = [
        [0.1, 0.2, 0.0, 0.1, 0.0, 0.0],
        [0.0, 4.0, 0.2, 0.0, 0.1, 0.0],
        [0.0, 0.1, 4.0, 0.0, 0.0, 0.1],
        [2.0, 0.0, 0.0, 5.0, 0.1, 0.0],
        [0.1, 0.0, 0.0, 0.2, 4.0, 0.5],
        [0.0, 0.1, 0.0, 0.0, 0.1, 3.0],
    ]
    a1 = [
        [5.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.2, 4.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.1, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    f0 = rap.factor_rap(a0)
    f1 = rap.factor_rap(a1)
    assert f0.valid and f1.valid and f0.pivots != tuple(range(6))
    if not graph_valid:
        f1 = rap.Factor(f1.lu, f1.pivots, False)

    n, k = 3, 2
    P = [[0.0, 0.0] for _ in range(3*n)]
    AP = [[0.0, 0.0] for _ in range(3*n)]
    P[0][0] = 1.0
    P[1][1] = 1.0
    AP[0] = [0.0, 2.0]
    AP[1] = [1.0, 3.0]
    AP[2] = [0.4, -0.1]
    AP[3] = [-0.2, 0.6]
    AP[4] = [0.3, 0.2]
    AP[5] = [0.1, -0.3]
    # Fixed node rows are deliberately zero in effective P and AP.
    R = [[0.0, 2.0], [1.0, 3.0]]
    rf = rap.factor_rap(R)
    assert rf.valid and rf.pivots == (1, 0)
    if not rap_valid:
        rf = rap.Factor(rf.lu, rf.pivots, False)

    # Packed membership indices: patch0 contributes entries 0,1; patch1 2,3.
    capture = g.GraphCapture(
        node_count=n, free_mask=(True, True, False),
        patches=(g.GraphPatch((0, 1), a0, f0),
                 g.GraphPatch((1, 2), a1, f1)),
        patch_nodes=(0, 1, 1, 2),
        node_patch_offsets=(0, 1, 3, 4),
        node_patch_values=(0, 1, 2, 3),
        node_coverage=(1, 2, 1),
        P=P, AP=AP, base_factor=rf, mode_stride=3,
        augmentation_enabled=augmentation, completed_tail_latch=tail_latch,
        identity=("physics", "program", 8, 0, 0, 0, 2, 0x3f800000),
    )
    g.validate_capture(capture)
    return capture


class Selector5GraphReferenceTests(unittest.TestCase):
    def test_overlapping_nonsymmetric_pivoted_map_and_transpose(self):
        c = fixture()
        self.assertEqual(c.node_coverage, (1, 2, 1))
        self.assertNotEqual(c.patches[0].factor.pivots, tuple(range(6)))
        self.assertEqual(c.base_factor.pivots, (1, 0))
        self.assertLess(g.factor_reconstruction_error(c.patches[0].original, c.patches[0].factor)["relative_to_max_abs"], 1e-12)
        self.assertLess(g.factor_reconstruction_error(c.patches[1].original, c.patches[1].factor)["relative_to_max_abs"], 1e-12)
        self.assertLess(g.factor_reconstruction_error(g.base_rap_matrix(c), c.base_factor)["relative_to_max_abs"], 1e-12)
        self.assertTrue(g.compare_base_rap_to_ptap(c, atol=1e-12)["within_atol"])
        x = [0.2, -0.4, 0.8, 0.1, -0.3, 0.5, 7.0, 8.0, -9.0]
        y = [-0.7, 0.9, 0.3, 0.6, -0.2, 0.1, 4.0, -5.0, 3.0]
        sx, sty = g.apply_base_S(c, x), g.apply_base_S_transpose(c, y)
        self.assertEqual(sx[6:], [0.0, 0.0, 0.0])
        self.assertEqual(sty[6:], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(rap.dot(sx, y), rap.dot(x, sty), delta=2e-12)

    def test_global_A_patch_principal_comparison_and_identity_gate(self):
        c = fixture()
        A = [[0.0] * 9 for _ in range(9)]
        # Create the global free-free block from patch0. Node1 principal block
        # is then exactly the block used by patch1; fixed node2 is identity.
        for i in range(6):
            for j in range(6):
                A[i][j] = c.patches[0].original[i][j]
        for i in range(6, 9):
            A[i][i] = 1.0
        result = g.compare_patch_matrices_to_global_A(
            c, A, identity_expected=c.identity, atol=1e-12)
        self.assertTrue(result["within_atol"])
        wrong_identity = tuple(c.identity[:-1]) + (123,)
        with self.assertRaisesRegex(g.GraphReferenceError, "identity"):
            g.compare_patch_matrices_to_global_A(c, A, identity_expected=wrong_identity)

    def test_invalid_local_factor_fails_closed_but_completed_tail_zeroes(self):
        invalid = fixture(graph_valid=False)
        with self.assertRaisesRegex(g.GraphReferenceError, "invalid graph patch factor"):
            g.apply_base_S(invalid, [1.0] + [0.0]*8)
        tail = fixture(graph_valid=False, tail_latch=True)
        self.assertEqual(g.apply_base_S(tail, [1.0] + [0.0]*8), [0.0]*9)
        self.assertEqual(g.apply_base_S_transpose(tail, [0.0, 1.0] + [0.0]*7), [0.0]*9)

    def test_invalid_base_rap_retains_graph_only_action_and_transpose(self):
        valid_factor_capture = fixture(rap_valid=True)
        invalid_factor_capture = fixture(rap_valid=False)
        x = [0.2, -0.4, 0.8, 0.1, -0.3, 0.5, 0.0, 0.0, 0.0]
        y = [-0.7, 0.9, 0.3, 0.6, -0.2, 0.1, 0.0, 0.0, 0.0]
        self.assertEqual(g.apply_base_S(invalid_factor_capture, x),
                         g.apply_H(invalid_factor_capture, x))
        self.assertEqual(g.apply_base_S_transpose(invalid_factor_capture, y),
                         g.apply_H_transpose(invalid_factor_capture, y))
        sx = g.apply_base_S(invalid_factor_capture, x)
        sty = g.apply_base_S_transpose(invalid_factor_capture, y)
        self.assertAlmostEqual(rap.dot(sx, y), rap.dot(x, sty), delta=2e-12)

    def test_fixed_mask_and_overlap_are_required_not_approximated(self):
        c = fixture()
        malformed = g.GraphCapture(**{**c.__dict__, "node_coverage": (1, 1, 1)})
        with self.assertRaisesRegex(g.GraphReferenceError, "coverage"):
            g.validate_capture(malformed)
        malformed = g.GraphCapture(**{**c.__dict__, "node_patch_values": (0, 2, 1, 3)})
        with self.assertRaisesRegex(g.GraphReferenceError, "CSR"):
            g.validate_capture(malformed)
        malformed_P = [row[:] for row in c.P]
        malformed_P[6][0] = 1.0
        malformed = g.GraphCapture(**{**c.__dict__, "P": malformed_P})
        with self.assertRaisesRegex(g.GraphReferenceError, "fixed"):
            g.validate_capture(malformed)

    def test_augmented_capture_is_base_reference_only_and_actual_request_rejects(self):
        c = fixture(augmentation=True)
        self.assertEqual(g.require_base_view(c),
                         "captured_base_view_reference_only_not_actual_callback_selection")
        with self.assertRaisesRegex(g.GraphReferenceError, "augmented consuming restart"):
            g.require_base_view(c, request_actual_consumed_view=True)
        c = fixture(augmentation=False)
        with self.assertRaisesRegex(g.GraphReferenceError, "does not prove"):
            g.require_base_view(c, request_actual_consumed_view=True)

    def test_nonfinite_rhs_rejected(self):
        c = fixture()
        with self.assertRaisesRegex(g.GraphReferenceError, "nonfinite"):
            g.apply_base_S(c, [float("nan")] + [0.0]*8)

    def test_source_layout_checksum_uses_canonical_word_order(self):
        value = g.fnv1a32_graph_layout(
            ((0, 2, 0, 0),), ((0, 0, 0, 0), (0, 1, 0, 1)),
            (0, 1), (0, 1, 2), (0, 1))
        self.assertEqual(value, 0x686E85AC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
