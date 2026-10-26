"""Read-only LP diagnostic for per-target no-regression rows.

All matrices are in the fitter's output displacement coordinates unless an
optional output_map M is supplied. This module does not alter solver policy,
geometry, acceptance gates, or persisted candidate state.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def _csr(value: Any, name: str) -> sparse.csr_matrix:
    out = sparse.csr_matrix(value, dtype=np.float64)
    if out.ndim != 2 or not np.isfinite(out.data).all():
        raise ValueError(f"{name} must be a finite 2-D matrix")
    out.sum_duplicates()
    out.sort_indices()
    return out


def _one_dim(value: Any, name: str, *, size: int | None = None,
             allow_infinite: bool = False) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 1 or (size is not None and out.shape != (size,)):
        raise ValueError(f"{name} must be a vector of length {size}" if size is not None
                         else f"{name} must be one-dimensional")
    valid = ~np.isnan(out) if allow_infinite else np.isfinite(out)
    if not valid.all():
        raise ValueError(f"{name} contains invalid values")
    return out


def _solve_max_target(A: sparse.csr_matrix, b: np.ndarray,
                      lower: np.ndarray, upper: np.ndarray,
                      M: sparse.csr_matrix, C: sparse.csr_matrix,
                      h: np.ndarray, target_row: int, *,
                      include_no_regression: bool,
                      time_limit_s: float, lp_method: str) -> dict:
    n_out, n_control = M.shape
    Am = (A @ M).tocsr()
    Cm = (C @ M).tocsr()
    lower_rows = np.flatnonzero(np.isfinite(lower))
    upper_rows = np.flatnonzero(np.isfinite(upper))
    blocks = [(-Am).tocsr(), (-M[lower_rows, :]).tocsr(), M[upper_rows, :].tocsr()]
    rhs = [-b, -lower[lower_rows], upper[upper_rows]]
    block_names = ["hard_A", "output_lower", "output_upper"]
    if include_no_regression:
        blocks.append((-Cm).tocsr())
        rhs.append(np.zeros(C.shape[0], dtype=np.float64))
        block_names.append("all_clearance_no_regression")
    aub = sparse.vstack(blocks, format="csr")
    bub = np.concatenate(rhs)

    target = Cm.getrow(target_row).toarray().reshape(-1) / h[target_row]
    objective = -target
    result = linprog(
        objective,
        A_ub=aub,
        b_ub=bub,
        bounds=[(None, None)] * n_control,
        method=lp_method,
        options={"time_limit": float(time_limit_s), "presolve": True},
    )
    out = {
        "include_all_no_regression_rows": bool(include_no_regression),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "solver": "scipy.optimize.linprog/HiGHS",
        "lp_method": str(lp_method),
        "constraint_blocks": block_names,
        "constraint_row_counts": [int(a.shape[0]) for a in blocks],
    }
    if not result.success:
        out["certificate_passed"] = False
        return out

    q = np.asarray(result.x, dtype=np.float64)
    d = np.asarray(M @ q, dtype=np.float64).reshape(-1)
    hard_slack = np.asarray(A @ d, dtype=np.float64).reshape(-1) - b
    no_regress = np.asarray(C @ d, dtype=np.float64).reshape(-1)
    target_progress = float(C.getrow(target_row).dot(d)[0])
    bound_lower_slack = d - lower
    bound_upper_slack = upper - d
    min_hard = float(hard_slack.min(initial=np.inf))
    min_lo = float(bound_lower_slack.min(initial=np.inf))
    min_hi = float(bound_upper_slack.min(initial=np.inf))
    min_no_regress = float(no_regress.min(initial=np.inf))
    certificate_passed = bool(
        min_hard >= -2e-7 and min_lo >= -2e-7 and min_hi >= -2e-7
        and (not include_no_regression or min_no_regress >= -2e-7)
    )
    out.update({
        "target_row": int(target_row),
        "target_h_mm": float(h[target_row]),
        "maximum_normalized_target_progress": float(target_progress / h[target_row]),
        "maximum_target_progress_mm": target_progress,
        "minimum_hard_A_slack": min_hard,
        "minimum_output_lower_slack": min_lo,
        "minimum_output_upper_slack": min_hi,
        "minimum_clearance_no_regression_slack_mm": min_no_regress,
        "certificate_passed": certificate_passed,
        "primal_delta_output": d.tolist(),
        "primal_control_coordinates": q.tolist(),
    })
    # Keep HiGHS dual information so an optimality result is inspectable.
    if getattr(result, "ineqlin", None) is not None:
        marg = getattr(result.ineqlin, "marginals", None)
        resid = getattr(result.ineqlin, "residual", None)
        out["inequality_dual_marginals"] = None if marg is None else np.asarray(marg, dtype=np.float64).tolist()
        out["inequality_primal_residuals"] = None if resid is None else np.asarray(resid, dtype=np.float64).tolist()
    if getattr(result, "lower", None) is not None:
        marg = getattr(result.lower, "marginals", None)
        out["control_lower_dual_marginals"] = None if marg is None else np.asarray(marg, dtype=np.float64).tolist()
    if getattr(result, "upper", None) is not None:
        marg = getattr(result.upper, "marginals", None)
        out["control_upper_dual_marginals"] = None if marg is None else np.asarray(marg, dtype=np.float64).tolist()
    return out


def compare_target_progress_feasibility(
    A: Any,
    b: Any,
    lower: Any,
    upper: Any,
    output_map: Any | None,
    C: Any,
    h: Any,
    target_row: int,
    *,
    time_limit_s: float = 30.0,
    lp_method: str = "highs",
) -> dict:
    """Compare target-row LP maxima with and without global C @ d >= 0.

    The objective in both LPs is max (C[target_row] @ d) / h[target_row].
    A and C act on output-space displacement d; when output_map M is given,
    d=Mq and all constraints/bounds are mapped exactly through M. The first
    LP includes A d >= b and output bounds only. The second adds every
    no-regression inequality C d >= 0. This is a feasibility diagnostic,
    not a proposed geometry or a change to the production objective/gates.
    """
    hard = _csr(A, "A")
    clearance = _csr(C, "C")
    n_out = hard.shape[1]
    if clearance.shape[1] != n_out:
        raise ValueError("A and C must share output-coordinate width")
    rhs = _one_dim(b, "b", size=hard.shape[0])
    lo = _one_dim(lower, "lower", size=n_out, allow_infinite=True)
    hi = _one_dim(upper, "upper", size=n_out, allow_infinite=True)
    if np.any(lo > hi):
        raise ValueError("lower output bounds exceed upper bounds")
    heights = _one_dim(h, "h", size=clearance.shape[0])
    if np.any(heights <= 0.0):
        raise ValueError("all h entries must be positive")
    if isinstance(target_row, (bool, np.bool_)) or not isinstance(target_row, (int, np.integer)):
        raise ValueError("target_row must be an integer clearance-row index")
    target_row = int(target_row)
    if not 0 <= target_row < clearance.shape[0]:
        raise ValueError("target_row is outside C")
    if not np.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("time_limit_s must be finite and positive")
    if lp_method not in {"highs", "highs-ipm", "highs-ds"}:
        raise ValueError("lp_method must be a supported HiGHS method")
    if output_map is None:
        M = sparse.eye(n_out, format="csr", dtype=np.float64)
    else:
        M = _csr(output_map, "output_map")
        if M.shape[0] != n_out or M.shape[1] == 0:
            raise ValueError("output_map must have shape (output_dimension, nonzero control_dimension)")
    without = _solve_max_target(
        hard, rhs, lo, hi, M, clearance, heights, target_row,
        include_no_regression=False, time_limit_s=float(time_limit_s), lp_method=lp_method,
    )
    with_all = _solve_max_target(
        hard, rhs, lo, hi, M, clearance, heights, target_row,
        include_no_regression=True, time_limit_s=float(time_limit_s), lp_method=lp_method,
    )
    decrease = None
    if without["success"] and with_all["success"]:
        decrease = (without["maximum_normalized_target_progress"]
                    - with_all["maximum_normalized_target_progress"])
    return {
        "schema": "numi.human.target-row-progress-feasibility.397.v1",
        "classification": "read-only LP feasibility diagnostic; not geometry, admission, or a production solver policy",
        "target_row": target_row,
        "target_h_mm": float(heights[target_row]),
        "output_dimension": int(n_out),
        "control_dimension": int(M.shape[1]),
        "hard_row_count": int(hard.shape[0]),
        "clearance_row_count": int(clearance.shape[0]),
        "results": {
            "without_global_no_regression": without,
            "with_global_no_regression": with_all,
        },
        "normalized_target_progress_reduction": decrease,
    }