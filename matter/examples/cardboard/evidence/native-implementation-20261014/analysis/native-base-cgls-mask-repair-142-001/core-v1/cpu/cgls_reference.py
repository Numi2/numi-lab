"""FP32 reference for the bounded CGLS vector recurrences.

This file models only the Krylov recurrence and strict full-vector masking. It
does not evaluate the production FEM operator, install a correction, or confer
the original FGMRES certificate on a CGLS candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy import sparse

RUNNING, EXHAUSTED, ZERO_NORMAL_RESIDUAL, BREAKDOWN, INVALID_INPUT = range(1, 6)


@dataclass
class Result:
    status: int
    iterations: int
    direction: np.ndarray
    residual: np.ndarray
    gradient: np.ndarray
    gamma: float
    alpha: float
    beta: float
    initial_rhs_norm: float
    history: list[dict[str, float]]
    error: str | None = None


def _compensated_add(sum_value: np.float32, correction: np.float32,
                     value: np.float32) -> tuple[np.float32, np.float32]:
    adjusted = np.float32(value - correction)
    next_value = np.float32(sum_value + adjusted)
    next_correction = np.float32(np.float32(next_value - sum_value) - adjusted)
    return next_value, next_correction


def _simd_sum(values: np.ndarray) -> np.float32:
    work = np.asarray(values, dtype=np.float32).copy()
    if work.shape != (32,):
        raise ValueError("reference reduction requires exactly 32 SIMD lanes")
    offset = 16
    while offset:
        for lane in range(offset):
            work[lane] = np.float32(work[lane] + work[lane + offset])
        offset //= 2
    return np.float32(work[0])


def _dot_full(left: np.ndarray, right: np.ndarray, mask: np.ndarray,
              environment_map: np.ndarray, selected: np.ndarray,
              env_count: int, node_count: int, env: int) -> np.float32:
    if int(selected[env]) == 0:
        return np.float32(0.0)
    lanes = np.zeros(32, dtype=np.float32)
    correction = np.zeros(32, dtype=np.float32)
    mechanical_rows = env_count * node_count
    for lane in range(32):
        for local in range(lane, 2 * node_count, 32):
            index = (env * node_count + local if local < node_count else
                     mechanical_rows + env * node_count + local - node_count)
            if int(environment_map[index]) != env:
                raise ValueError("row environment map mismatch")
            if local >= node_count:
                continue
            m = mask[index, :3]
            if np.any((m != 0.0) & (m != 1.0)):
                raise ValueError("row mask is not exact 0/1")
            a = np.asarray(left[index, :3] * m, dtype=np.float32)
            b = np.asarray(right[index, :3] * m, dtype=np.float32)
            # Match the scalar component ordering of dot(float3,float3).
            term = np.float32(np.float32(a[0] * b[0]) +
                              np.float32(a[1] * b[1]))
            term = np.float32(term + np.float32(a[2] * b[2]))
            lanes[lane], correction[lane] = _compensated_add(
                lanes[lane], correction[lane], term)
    return _simd_sum(lanes)


def _apply(matrix: sparse.csr_matrix, vector: np.ndarray, free_nodes: np.ndarray,
           node_count: int, env_count: int) -> np.ndarray:
    compressed = np.empty(matrix.shape[1], dtype=np.float32)
    for free_index, node in enumerate(free_nodes):
        for axis in range(3):
            compressed[3 * free_index + axis] = vector[int(node), axis]
    result = np.zeros(matrix.shape[0], dtype=np.float32)
    for row in range(matrix.shape[0]):
        total = np.float32(0.0)
        for cursor in range(matrix.indptr[row], matrix.indptr[row + 1]):
            product = np.float32(matrix.data[cursor] * compressed[matrix.indices[cursor]])
            total = np.float32(total + product)
        result[row] = total
    output = np.zeros((node_count, 4), dtype=np.float32)
    for free_index, node in enumerate(free_nodes):
        output[int(node), :3] = result[3 * free_index:3 * free_index + 3]
    return output


def run_cgls_fp32(matrix: sparse.csr_matrix, rhs: np.ndarray,
                  free_nodes: np.ndarray, *, node_count: int,
                  env_count: int = 2, selected: tuple[int, ...] = (1, 0),
                  cap: int = 32, row_mask: np.ndarray | None = None,
                  environment_map: np.ndarray | None = None) -> Result:
    """Run the exact unregularized CGLS recurrence in masked full-vector form."""
    A = sparse.csr_matrix(matrix, dtype=np.float32)
    A.sum_duplicates()
    A.sort_indices()
    At = A.transpose().tocsr()
    At.sort_indices()
    if cap not in (32, 64):
        return Result(INVALID_INPUT, 0, np.zeros((0, 4), np.float32),
                      np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32),
                      0.0, 0.0, 0.0, 0.0, [], "iteration cap must be 32 or 64")
    if env_count <= 0 or node_count <= 0 or A.shape != (3 * len(free_nodes),) * 2:
        return Result(INVALID_INPUT, 0, np.zeros((0, 4), np.float32),
                      np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32),
                      0.0, 0.0, 0.0, 0.0, [], "matrix/layout mismatch")
    U = 2 * env_count * node_count
    expected = np.asarray(selected, dtype=np.uint32)
    if expected.shape != (env_count,) or np.any(expected > 1):
        return Result(INVALID_INPUT, 0, np.zeros((0, 4), np.float32),
                      np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32),
                      0.0, 0.0, 0.0, 0.0, [], "invalid selected-environment map")
    mask = np.zeros((U, 4), np.float32) if row_mask is None else np.asarray(row_mask, dtype=np.float32)
    row_env = np.empty(U, dtype=np.uint32) if environment_map is None else np.asarray(environment_map, dtype=np.uint32)
    if mask.shape != (U, 4) or row_env.shape != (U,):
        return Result(INVALID_INPUT, 0, np.zeros((0, 4), np.float32),
                      np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32),
                      0.0, 0.0, 0.0, 0.0, [], "mask/environment map has wrong shape")
    if row_mask is None:
        for e, is_selected in enumerate(expected):
            if is_selected:
                for node in free_nodes:
                    mask[e * node_count + int(node), :3] = 1.0
    if environment_map is None:
        half = env_count * node_count
        for i in range(U):
            local = i if i < half else i - half
            row_env[i] = local // node_count

    rhs_array = np.asarray(rhs, dtype=np.float32)
    if rhs_array.shape == (U, 4):
        b = rhs_array.copy()
        if not np.isfinite(b).all() or np.any(b[:, 3] != 0.0):
            return Result(INVALID_INPUT, 0, b, b.copy(), b.copy(), 0.0, 0.0,
                          0.0, 0.0, [], "full RHS contains nonfinite values or nonzero w lanes")
    else:
        b_free = rhs_array.reshape(-1)
        if b_free.size != A.shape[0] or not np.isfinite(b_free).all():
            return Result(INVALID_INPUT, 0, np.zeros((U, 4), np.float32),
                          np.zeros((U, 4), np.float32), np.zeros((U, 4), np.float32),
                          0.0, 0.0, 0.0, 0.0, [], "RHS invalid")
        b = np.zeros((U, 4), np.float32)
        for free_index, node in enumerate(free_nodes):
            b[int(node), :3] = b_free[3 * free_index:3 * free_index + 3]
    # This recurrence reference uses one selected environment. The second
    # environment is retained as an exact quiet twin to exercise layout.
    if tuple(int(x) for x in expected) != (1, 0):
        return Result(INVALID_INPUT, 0, b, b.copy(), b.copy(), 0.0, 0.0, 0.0,
                      0.0, [], "reference bundle currently requires selected env0 and quiet env1")

    mechanical_rows = env_count * node_count
    for i in range(U):
        e = int(row_env[i])
        if e >= env_count:
            return Result(INVALID_INPUT, 0, b, b.copy(), b.copy(), 0.0, 0.0, 0.0,
                          0.0, [], "row environment out of range")
        expected_e = (i if i < mechanical_rows else i - mechanical_rows) // node_count
        if e != expected_e:
            return Result(INVALID_INPUT, 0, b, b.copy(), b.copy(), 0.0, 0.0, 0.0,
                          0.0, [], "row environment map mismatch")
        if not np.isfinite(mask[i]).all() or mask[i, 3] != 0.0 or \
                np.any((mask[i, :3] != 0.0) & (mask[i, :3] != 1.0)):
            return Result(INVALID_INPUT, 0, b, b.copy(), b.copy(), 0.0, 0.0, 0.0,
                          0.0, [], "invalid exact free/fixed mask")
        if i >= mechanical_rows and (np.any(mask[i] != 0.0) or np.any(b[i] != 0.0)):
            return Result(INVALID_INPUT, 0, b, b.copy(), b.copy(), 0.0, 0.0, 0.0,
                          0.0, [], "field backing rows must be exactly zero")
        if e == 1 and (np.any(mask[i] != 0.0) or np.any(b[i] != 0.0)):
            return Result(INVALID_INPUT, 0, b, b.copy(), b.copy(), 0.0, 0.0, 0.0,
                          0.0, [], "quiet environment must be exactly zero")
        if i < mechanical_rows and np.any((mask[i, :3] == 0.0) & (b[i, :3] != 0.0)):
            return Result(INVALID_INPUT, 0, b, b.copy(), b.copy(), 0.0, 0.0, 0.0,
                          0.0, [], "fixed rows must have zero RHS")

    x = np.zeros((U, 4), np.float32)
    r = b.copy()
    s = np.zeros_like(b)
    s[:node_count] = _apply(At, b[:node_count], free_nodes, node_count, env_count)
    p = s.copy()
    q = np.zeros_like(b)
    history: list[dict[str, float]] = []
    gamma = _dot_full(s, s, mask, row_env, expected, env_count, node_count, 0)
    bnorm2 = _dot_full(b, b, mask, row_env, expected, env_count, node_count, 0)
    alpha = beta = np.float32(0.0)
    if not math.isfinite(float(gamma)) or not math.isfinite(float(bnorm2)):
        status = INVALID_INPUT
        err = "nonfinite initial reduction"
    elif gamma == 0.0 or bnorm2 == 0.0:
        status = ZERO_NORMAL_RESIDUAL
        err = None
    else:
        status = RUNNING
        err = None
    iterations = 0
    for _ in range(cap):
        if status != RUNNING:
            break
        q[:node_count] = _apply(A, p[:node_count], free_nodes, node_count, env_count)
        qNorm2 = _dot_full(q, q, mask, row_env, expected, env_count, node_count, 0)
        if not math.isfinite(float(qNorm2)) or qNorm2 <= 0.0 or gamma <= 0.0:
            status, err = BREAKDOWN, "nonpositive/nonfinite p^T A^T A p"
            break
        alpha = np.float32(gamma / qNorm2)
        if not math.isfinite(float(alpha)) or alpha <= 0.0:
            status, err = BREAKDOWN, "nonpositive/nonfinite alpha"
            break
        x[:node_count, :3] = (x[:node_count, :3] + alpha * p[:node_count, :3]) * mask[:node_count, :3]
        r[:node_count, :3] = (r[:node_count, :3] - alpha * q[:node_count, :3]) * mask[:node_count, :3]
        x[:, 3] = 0.0
        r[:, 3] = 0.0
        iterations += 1
        s[:] = 0.0
        s[:node_count] = _apply(At, r[:node_count], free_nodes, node_count, env_count)
        next_gamma = _dot_full(s, s, mask, row_env, expected, env_count, node_count, 0)
        history.append({"iteration": float(iterations), "alpha": float(alpha),
                        "gamma": float(gamma), "next_gamma": float(next_gamma),
                        "residual_norm": float(np.sqrt(max(float(_dot_full(
                            r, r, mask, row_env, expected, env_count,
                            node_count, 0)), 0.0)))})
        if not math.isfinite(float(next_gamma)) or next_gamma < 0.0:
            status, err = BREAKDOWN, "nonfinite next normal residual"
            break
        if next_gamma == 0.0:
            gamma = np.float32(0.0)
            status = ZERO_NORMAL_RESIDUAL
            break
        beta = np.float32(next_gamma / gamma)
        if not math.isfinite(float(beta)) or beta <= 0.0:
            status, err = BREAKDOWN, "nonpositive/nonfinite beta"
            break
        gamma = next_gamma
        if iterations == cap:
            status = EXHAUSTED
            break
        p[:node_count, :3] = (s[:node_count, :3] + beta * p[:node_count, :3]) * mask[:node_count, :3]
        p[:, 3] = 0.0
    if status == RUNNING:
        status = EXHAUSTED if iterations >= cap else BREAKDOWN
    return Result(int(status), iterations, x, r, s, float(gamma), float(alpha),
                  float(beta), float(bnorm2), history, err)


def make_layout(rhs_free: np.ndarray, free_nodes: np.ndarray, node_count: int,
                env_count: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create exact (E,N) mechanical + zero field float4 backing rows."""
    unknown_count = 2 * env_count * node_count
    vectors = np.zeros((unknown_count, 4), np.float32)
    mask = np.zeros_like(vectors)
    row_env = np.empty(unknown_count, dtype=np.uint32)
    for index in range(unknown_count):
        half = env_count * node_count
        local = index if index < half else index - half
        row_env[index] = local // node_count
    rhs_values = np.asarray(rhs_free, np.float32).reshape(-1)
    if rhs_values.size != 3 * len(free_nodes):
        raise ValueError("free RHS shape mismatch")
    for j, node in enumerate(free_nodes):
        vectors[int(node), :3] = rhs_values[3*j:3*j+3]
        mask[int(node), :3] = 1.0
    return vectors, mask, row_env
