"""Independent small algebra/guard checks against the sealed 158 implementation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
from scipy.sparse import csr_matrix

BASE = Path('/Users/home/cardboard-implementation-20261014')
RUN = BASE / 'analysis/cgls-node-block-rap-reference-158-001/run.py'
spec = importlib.util.spec_from_file_location('sealed_run158', RUN)
run158 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run158
spec.loader.exec_module(run158)


def no_pivot_lu(a: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build the exact final-row-permutation storage expected by solve_lu."""
    n = len(a)
    lu = np.array(a, dtype=np.float64, copy=True)
    for k in range(n):
        if lu[k, k] == 0.0:
            raise AssertionError('synthetic coarse matrix unexpectedly needs pivoting')
        for i in range(k + 1, n):
            lu[i, k] /= lu[k, k]
            for j in range(k + 1, n):
                lu[i, j] -= lu[i, k] * lu[k, j]
    return lu, tuple(range(n))


def test_nonsymmetric_map_and_exact_algebraic_transpose() -> dict:
    # Two 3-DOF nodes, nonsymmetric A and R, nontrivial P, and off-diagonal
    # coupling. This exercises the actual sealed BlockRAP.apply/transpose code.
    A = np.array([
        [4.0, .4, .1, .2, .0, .1],
        [-.2, 3.5, .3, .0, .1, .0],
        [.0, -.1, 2.8, .1, .0, .2],
        [.3, .0, -.1, 3.7, .2, .0],
        [.0, .2, .0, -.1, 4.2, .4],
        [.1, .0, .2, .0, -.3, 3.1],
    ], dtype=np.float64)
    P = np.array([
        [1.0, .1], [.0, 1.0], [.2, .0],
        [.1, .3], [.5, .2], [.0, 1.0],
    ], dtype=np.float64)
    AP = A @ P
    R = P.T @ AP
    if np.linalg.norm(R - R.T) < 1e-3:
        raise AssertionError('synthetic RAP must be nonsymmetric')
    lu, piv = no_pivot_lu(R)
    cap = SimpleNamespace(
        node_count=2,
        P=P,
        AP=AP,
        base_factor=SimpleNamespace(valid=True, lu=lu, pivots=piv),
    )
    sparse_A = csr_matrix(A)
    errors = []
    for weight in (.1, 1.0):
        mapping = run158.BlockRAP(sparse_A, [0, 1], cap, np.float64, weight)
        H = np.zeros_like(A)
        H[:3, :3] = np.linalg.inv(A[:3, :3])
        H[3:, 3:] = np.linalg.inv(A[3:, 3:])
        Q = P - H @ AP
        S = H + weight * Q @ np.linalg.solve(R, P.T)
        v = np.array([.3, -.5, .9, .2, 1.1, -.7], dtype=np.float64)
        u = np.array([-.4, .8, .2, 1.0, -.3, .6], dtype=np.float64)
        direct = mapping.apply(v)
        direct_t = mapping.transpose(u)
        err_s = float(np.max(np.abs(direct - S @ v)))
        err_st = float(np.max(np.abs(direct_t - S.T @ u)))
        dot_gap = float(abs(u @ direct - direct_t @ v))
        if err_s > 2e-13 or err_st > 2e-13 or dot_gap > 2e-13:
            raise AssertionError(f'nonsymmetric S/S^T mismatch at weight={weight}: {err_s}, {err_st}, {dot_gap}')
        errors.append({'weight': weight, 'S_max_abs': err_s, 'ST_max_abs': err_st,
                       'adjoint_dot_abs': dot_gap})
    return {'result': 'pass', 'nonsymmetric_R': True, 'cases': errors}


def test_invalid_factor_rejected_by_actual_reference_solve() -> dict:
    bad = SimpleNamespace(valid=False, lu=np.eye(2), pivots=(0, 1))
    try:
        run158.ref.solve_lu(bad, np.array([1.0, 2.0]), np.float64, False)
    except run158.ref.ExperimentError as exc:
        if str(exc) != 'base RAP factor invalid':
            raise AssertionError(f'unexpected invalid-factor error: {exc}') from exc
    else:
        raise AssertionError('source reference solve accepted an invalid RAP factor')
    # Also exercise the sealed wrapper's admission guard with a captured-like
    # object; the study itself asserts this flag on its real captured factor.
    invalid_capture = SimpleNamespace(node_count=2, P=np.ones((6, 2)),
                                      AP=np.ones((6, 2)), base_factor=bad)
    try:
        run158.BlockRAP(csr_matrix(np.eye(6)), [0, 1], invalid_capture, np.float64, .1)
    except AssertionError:
        pass
    else:
        raise AssertionError('BlockRAP admitted invalid captured factor')
    return {'result': 'pass', 'direct_solve_rejects_invalid_factor': True,
            'study_wrapper_rejects_invalid_factor': True,
            'wrapper_guard_kind': 'Python assert (removed by python -O)'}


if __name__ == '__main__':
    report = {'schema': 'numi.cardboard.block_rap_review_tests.v1',
              'tests': [test_nonsymmetric_map_and_exact_algebraic_transpose(),
                        test_invalid_factor_rejected_by_actual_reference_solve()]}
    print(__import__('json').dumps(report, indent=2, allow_nan=False))
