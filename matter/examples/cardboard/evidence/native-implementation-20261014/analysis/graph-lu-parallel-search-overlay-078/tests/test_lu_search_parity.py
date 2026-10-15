#!/usr/bin/env python3
"""Independent FP32 control for the graph-patch LU search rewrite.

This test mirrors only the pivot/scale selection and row swaps. The trailing
FP32 division/update operations are held identical in both paths. It is not a
Metal execution or solver qualification.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

BASE = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-stable-free-graph-and-general-arithmetic-076")
OVERLAY = Path("/Users/home/cardboard-implementation-20261014/analysis/graph-lu-parallel-search-overlay-078/overlay")


def f32(x: float) -> np.float32:
    return np.float32(x)


def lu(matrix: np.ndarray, parallel_search: bool) -> tuple[np.ndarray, list[int], bool, tuple[float, float, float]]:
    a = np.asarray(matrix, dtype=np.float32).copy()
    n = int(a.shape[0])
    piv = list(range(n))
    scale = f32(0.0)
    valid = True
    if parallel_search:
        local = []
        bad = False
        for lane in range(256):
            maximum = f32(0.0)
            for index in range(lane, n*n, 256):
                value = a.flat[index]
                if not np.isfinite(value):
                    bad = True
                else:
                    maximum = f32(max(maximum, f32(abs(value))))
            local.append(maximum)
        if bad:
            # Exactly the existing ordered scan's diagnostic prefix.
            scale = f32(0.0)
            for value in a.flat:
                if not np.isfinite(value):
                    valid = False
                    break
                scale = f32(max(scale, f32(abs(value))))
        else:
            for value in local:
                scale = f32(max(scale, value))
    else:
        for value in a.flat:
            if not np.isfinite(value):
                valid = False
                break
            scale = f32(max(scale, f32(abs(value))))
    if not (scale > 0.0 and np.isfinite(scale)):
        valid = False
    if not valid:
        return a, piv, False, (float(scale), math.nan, 5.0)

    floor = f32(max(f32(scale * f32(1.0e-10)), f32(1.0e-30)))
    last_pivot = f32(0.0)
    for k in range(n):
        bad = False
        if parallel_search:
            partials: list[tuple[np.float32, int]] = []
            for lane in range(256):
                maximum = f32(0.0)
                row_choice = k
                for row in range(k + lane, n, 256):
                    magnitude = f32(abs(a[row, k]))
                    if not np.isfinite(magnitude):
                        bad = True
                    elif magnitude > maximum:
                        maximum = magnitude
                        row_choice = row
                partials.append((maximum, row_choice))
            if bad:
                # Match lane-zero early break and partial magnitude.
                pivot_row = k
                pivot = f32(0.0)
                for row in range(k, n):
                    magnitude = f32(abs(a[row, k]))
                    if not np.isfinite(magnitude):
                        valid = False
                        break
                    if magnitude > pivot:
                        pivot, pivot_row = magnitude, row
            else:
                pivot_row = k
                pivot = f32(0.0)
                for magnitude, row in partials:
                    if magnitude > pivot or (magnitude == pivot and row < pivot_row):
                        pivot, pivot_row = magnitude, row
        else:
            pivot_row = k
            pivot = f32(0.0)
            for row in range(k, n):
                magnitude = f32(abs(a[row, k]))
                if not np.isfinite(magnitude):
                    valid = False
                    break
                if magnitude > pivot:
                    pivot, pivot_row = magnitude, row
        last_pivot = pivot
        if not (pivot > floor):
            valid = False
        if not valid:
            return a, piv, False, (float(scale), float(last_pivot), 6.0)
        if pivot_row != k:
            # Both paths perform the same disjoint element swap; the Metal
            # optimized path assigns columns to different group lanes.
            a[[k, pivot_row], :] = a[[pivot_row, k], :]
            piv[k], piv[pivot_row] = piv[pivot_row], piv[k]
        diagonal = a[k, k]
        if not np.isfinite(diagonal) or not (abs(diagonal) > floor):
            return a, piv, False, (float(scale), float(last_pivot), 6.0)
        width = n-k-1
        for offset in range(width):
            row = k+1+offset
            a[row, k] = f32(a[row, k] / diagonal)
        for offset in range(width*width):
            row = k+1+offset//width
            column = k+1+offset%width
            product = f32(a[row, k] * a[k, column])
            a[row, column] = f32(a[row, column] - product)
            if not np.isfinite(a[row, column]):
                return a, piv, False, (float(scale), float(last_pivot), 6.0)
    return a, piv, True, (float(scale), float(last_pivot), 0.0)


def bits(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype=np.float32).view(np.uint32).tobytes()


class GraphLUSearchParityTests(unittest.TestCase):
    def compare(self, matrix: np.ndarray):
        serial = lu(matrix, False)
        parallel = lu(matrix, True)
        self.assertEqual(bits(serial[0]), bits(parallel[0]))
        self.assertEqual(serial[1], parallel[1])
        self.assertEqual(serial[2], parallel[2])
        for a, b in zip(serial[3], parallel[3]):
            if math.isnan(a):
                self.assertTrue(math.isnan(b))
            else:
                self.assertEqual(np.float32(a).view(np.uint32),
                                 np.float32(b).view(np.uint32))
        return serial

    def test_seeded_finite_matrices_and_patch_sizes(self):
        rng = np.random.default_rng(781)
        for n in (3, 6, 12, 24, 48, 96):
            for _ in range(8):
                matrix = rng.normal(size=(n,n)).astype(np.float32)
                matrix += np.float32(n) * np.eye(n, dtype=np.float32)
                self.compare(matrix)

    def test_exact_tie_uses_first_candidate_row(self):
        matrix = np.asarray([[0, 1, 0], [2, 4, 1], [-2, 0, 3]], dtype=np.float32)
        result = self.compare(matrix)
        self.assertTrue(result[2])
        self.assertEqual(result[1][0], 1)

    def test_zero_and_singular_matrix_failure_diagnostics(self):
        self.assertFalse(self.compare(np.zeros((6,6), dtype=np.float32))[2])
        singular = np.asarray([[1,1,1],[1,1,1],[0,0,0]], dtype=np.float32)
        self.assertFalse(self.compare(singular)[2])

    def test_nonfinite_scale_scan_preserves_partial_diagnostic(self):
        for bad in (np.float32(np.nan), np.float32(np.inf), np.float32(-np.inf)):
            matrix = np.eye(6, dtype=np.float32)
            matrix[0,0] = 2.0
            matrix.flat[5] = bad
            result = self.compare(matrix)
            self.assertFalse(result[2])
            self.assertEqual(result[3][2], 5.0)
            self.assertEqual(np.float32(result[3][0]), np.float32(2.0))

    def test_pivot_floor_boundary(self):
        floor = np.float32(1.0e-10)
        for diagonal, expected in ((floor, False),
                                  (np.nextafter(floor, np.float32(np.inf)), True)):
            matrix = np.diag(np.asarray([1.0, 0.5, diagonal], dtype=np.float32))
            result = self.compare(matrix)
            self.assertEqual(result[2], expected)

    def test_trailing_overflow_keeps_same_failure(self):
        maximum = np.float32(3.0e38)
        matrix = np.asarray([[maximum, maximum, maximum],
                             [maximum, maximum, maximum],
                             [0.0, maximum, maximum]], dtype=np.float32)
        self.assertFalse(self.compare(matrix)[2])

    def test_source_keeps_update_arithmetic_unchanged_and_runtime_defaults_legacy(self):
        base = (BASE / "matter/src/metal/fgmres.metalinc").read_text()
        overlay = (OVERLAY / "matter/src/metal/fgmres.metalinc").read_text()
        marker = "        const uint width = lowerCount;"
        self.assertEqual(base[base.index(marker):base.index("// Add one signed product", base.index(marker))],
                         overlay[overlay.index(marker):overlay.index("// Add one signed product", overlay.index(marker))])
        runtime = (OVERLAY / "matter/src/runtime.mm").read_text()
        self.assertIn("const std::uint32_t parallelGraphPivotSearch = 0u;", runtime)
        self.assertIn("atIndex:13u", runtime)
        self.assertIn("constant uint& parallelSearch [[buffer(13)]]", overlay)
        checker = (Path(__file__).parents[1] / "checker/selector5_refinement_kernels.mm").read_text()
        self.assertIn("header, 0u);", checker)
        self.assertIn("header, 1u);", checker)
        self.assertIn("setCount(encoder, parallelSearch, 13u);", checker)


if __name__ == "__main__":
    unittest.main()
