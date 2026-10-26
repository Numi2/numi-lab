"""Pinned 408 reduced-basis LP copy without artificial soft-target no-regression rows."""
from __future__ import annotations
import hashlib
from functools import partial
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
EXPECTED_OWNER_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py")
EXPECTED_OWNER_SHA256 = "5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1 << 20),b""): h.update(block)
    return h.hexdigest()

def _solve_sparse_l1_clearance_in_basis_without_soft_noregression(owner, A: sparse.csr_matrix, lower_rows: np.ndarray,
                                       target_output: np.ndarray, lower_output: np.ndarray,
                                       upper_output: np.ndarray, C: sparse.csr_matrix,
                                       h: np.ndarray, weights: np.ndarray,
                                       output_map: sparse.csr_matrix,
                                       time_limit_s: float, slack_tolerance: float = 1e-9,
                                       allow_stage1_feasible_timeout_fallback: bool = False,
                                       lp_method: str = "highs",
                                       retained_gain_fraction: float | None = None) -> dict:
    """Solve the same lexicographic LP in reduced coordinates, preserving output L1 exactly.

    If d=Mq, every original hard/clearance row is transformed by M. Output
    trust/source bounds are also imposed as Mq lower/upper inequalities. The
    stage-2 absolute auxiliaries remain in output space, so the objective is
    exactly sum(abs(Mq-target_output)), not a parent-space surrogate.
    """
    lp_method = owner.validate_lp_method(lp_method)
    A = sparse.csr_matrix(A, dtype=np.float64)
    C = sparse.csr_matrix(C, dtype=np.float64)
    M = sparse.csr_matrix(output_map, dtype=np.float64)
    b = np.asarray(lower_rows, dtype=np.float64).reshape(-1)
    t = np.asarray(target_output, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower_output, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper_output, dtype=np.float64).reshape(-1)
    hh = np.asarray(h, dtype=np.float64).reshape(-1)
    ww = np.asarray(weights, dtype=np.float64).reshape(-1)
    n_out, n_control = M.shape
    m = len(hh)
    if (A.shape != (len(b), n_out) or C.shape != (m, n_out)
            or t.shape != (n_out,) or lo.shape != (n_out,) or hi.shape != (n_out,)
            or ww.shape != (m,) or n_control == 0):
        raise ValueError("basis clearance LP dimensions do not align")
    if not all(np.isfinite(x).all() for x in (A.data, C.data, M.data, b, t, lo, hi, hh, ww)):
        raise ValueError("basis clearance LP inputs contain nonfinite values")
    if np.any(lo > hi) or np.any(hh <= 0) or np.any(ww < 0) or (m and float(ww.sum()) <= 0):
        raise ValueError("basis clearance LP has invalid output bounds, targets, or weights")
    A_map = (A @ M).tocsr()
    hard = sparse.vstack((A_map, M, -M), format="csr")
    hard_rhs = np.concatenate((b, lo, -hi))
    hard, hard_rhs, hard_receipt = owner._drop_zero_hard_rows(hard, hard_rhs, "transformed hard/bound rows")
    C_map = (C @ M).tocsr()
    c_norm = np.sqrt(np.asarray(C_map.multiply(C_map).sum(axis=1)).reshape(-1))
    c_scale = np.ones(m, dtype=np.float64)
    nonzero_c = c_norm > 0.0
    c_scale[nonzero_c] = 1.0 / c_norm[nonzero_c]
    C_map = (sparse.diags(c_scale) @ C_map).tocsr()
    h_map = hh * c_scale
    started = time.monotonic()

    def certify_q(q, slack, objective):
        d = np.asarray(M @ q).reshape(-1)
        hard_residual = np.asarray(A @ d).reshape(-1) - b
        progress = np.asarray(C @ d).reshape(-1)
        minimum_progress = float(np.min(progress)) if progress.size else 0.0
        negative_progress_rows = int(np.count_nonzero(progress < -2e-7))
        slack_residual = progress + hh * slack - hh
        minimum_target_residual = float(np.min(slack_residual)) if slack_residual.size else 0.0
        minimum_hard_residual = float(np.min(hard_residual)) if hard_residual.size else 0.0
        bound_error = max(
            float(np.max(np.maximum(lo - d, 0.0), initial=0.0)),
            float(np.max(np.maximum(d - hi, 0.0), initial=0.0)),
            float(np.max(np.maximum(-slack, 0.0), initial=0.0)),
        )
        weighted = float(np.dot(ww, slack))
        result = {
            "passed": bool(
                np.min(hard_residual, initial=np.inf) >= -2e-7
                and np.min(slack_residual, initial=np.inf) >= -2e-7
                and bound_error <= 2e-7
                and abs(weighted - float(objective)) <= max(2e-7, abs(float(objective)) * 1e-9)
            ),
            "checked_hard_rows": int(A.shape[0]),
            "checked_output_bounds": int(2 * n_out),
            "checked_no_regression_rows": 0,
            "checked_target_rows": int(C.shape[0]),
            "checked_slack_bounds": int(m),
            "minimum_hard_residual": minimum_hard_residual,
            "minimum_soft_target_progress_mm": minimum_progress,
            "negative_soft_target_progress_rows": negative_progress_rows,
            "soft_target_regression_allowed": True,
            "no_regression_rows_enforced": 0,
            "minimum_target_residual": minimum_target_residual,
            "maximum_output_bound_violation": bound_error,
            "weighted_slack_objective": weighted,
            "reported_stage1_objective": float(objective),
        }
        return result, d

    # Stage 1: minimum normalized clearance slack.
    hard1 = sparse.hstack((-hard, sparse.csr_matrix((hard.shape[0], m))), format="csr")
    # C/h rows are soft targets. Their no-regression rows are omitted.
    # Physical hard rows remain in A, and output bounds remain in hard.
    clear1 = sparse.hstack((-C_map, -sparse.diags(h_map)), format="csr")
    aub1 = sparse.vstack((hard1, clear1), format="csr")
    bub1 = np.concatenate((-hard_rhs, -h_map))
    c1 = np.concatenate((np.zeros(n_control), ww))
    bounds1 = [(None, None)] * n_control + [(0.0, None)] * m
    stage1 = linprog(c1, A_ub=aub1, b_ub=bub1, bounds=bounds1, method=lp_method,
                     options={"time_limit": float(time_limit_s), "presolve": True})
    if not stage1.success:
        return {"success": False, "status": int(stage1.status), "message": str(stage1.message),
                "nit": int(getattr(stage1, "nit", 0)), "objective": None,
                "wall_seconds": float(time.monotonic() - started), "lp_method": lp_method,
                "stage1": {"success": False, "status": int(stage1.status), "message": str(stage1.message)},
                "basis": {"output_dimension": n_out, "control_dimension": n_control,
                          "hard_row_transform": hard_receipt}}
    q1 = np.asarray(stage1.x[:n_control], dtype=np.float64)
    s1 = np.asarray(stage1.x[n_control:], dtype=np.float64)
    cert, d1 = certify_q(q1, s1, float(stage1.fun))
    if not cert["passed"]:
        return {"success": False, "status": int(stage1.status),
                "message": "stage-1 reduced-coordinate primal failed output-space certification",
                "nit": int(getattr(stage1, "nit", 0)), "objective": None,
                "wall_seconds": float(time.monotonic() - started), "lp_method": lp_method,
                "stage1": {"success": False, "status": int(stage1.status), "objective": float(stage1.fun)},
                "stage1_output_certificate": cert}
    slack_opt = float(stage1.fun)
    tol = float(slack_tolerance)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("slack tolerance must be finite and nonnegative")
    budget_receipt = owner.stage2_retained_gain_budget(
        slack_opt, ww, tol, A, b, lo, hi, retained_gain_fraction=retained_gain_fraction
    )
    tol = float(budget_receipt["effective_slack_increase"])

    # Stage 2: lexicographically retain stage-1 clearance within tol, then
    # minimize the exact original output-space L1 distance.
    Znm = sparse.csr_matrix((n_out, m))
    Zmm = sparse.csr_matrix((m, m))
    hard2 = sparse.hstack((-hard, sparse.csr_matrix((hard.shape[0], n_out + m))), format="csr")
    # Keep target-slack inequalities but permit negative soft-target progress.
    clear2 = sparse.hstack((-C_map, sparse.csr_matrix((m, n_out)), -sparse.diags(h_map)), format="csr")
    abs_pos = sparse.hstack((M, -sparse.eye(n_out, format="csr"), Znm), format="csr")
    abs_neg = sparse.hstack((-M, -sparse.eye(n_out, format="csr"), Znm), format="csr")
    slack_budget = sparse.hstack((sparse.csr_matrix((1, n_control + n_out)), sparse.csr_matrix(ww.reshape(1, -1))), format="csr")
    aub2 = sparse.vstack((hard2, clear2, abs_pos, abs_neg, slack_budget), format="csr")
    bub2 = np.concatenate((-hard_rhs, -h_map, t, -t, [slack_opt + tol]))
    c2 = np.concatenate((np.zeros(n_control), np.ones(n_out), np.zeros(m)))
    bounds2 = [(None, None)] * n_control + [(0.0, None)] * n_out + [(0.0, None)] * m
    stage2 = linprog(c2, A_ub=aub2, b_ub=bub2, bounds=bounds2, method=lp_method,
                     options={"time_limit": float(time_limit_s), "presolve": True})
    if not stage2.success:
        status, message = int(stage2.status), str(stage2.message)
        time_limit = status == 1 and "time limit" in message.lower()
        fallback_cert = cert
        if allow_stage1_feasible_timeout_fallback and time_limit and cert["passed"]:
            return {
                "success": True, "status": int(stage1.status),
                "message": "Stage 2 hit its time limit; using independently certified feasible Stage 1 primal (no output-L1 tie-break optimality claim)",
                "nit": int(getattr(stage1, "nit", 0) + getattr(stage2, "nit", 0)),
                "objective": None, "wall_seconds": float(time.monotonic() - started),
                "lp_method": lp_method, "delta": d1, "basis_coordinates": q1,
                "normalized_slack": s1, "normalized_slack_objective": slack_opt,
                "stage1": {"success": True, "status": int(stage1.status), "objective": slack_opt},
                "stage2": {"success": False, "status": status, "message": message,
                           "time_limit": True},
                "stage1_output_certificate": fallback_cert,
                "minimum_soft_target_progress_mm": fallback_cert["minimum_soft_target_progress_mm"],
                "negative_soft_target_progress_rows": fallback_cert["negative_soft_target_progress_rows"],
                "soft_target_regression_allowed": True,
                "stage1_primal_certificate": {
                    "status": "CERTIFIED_STAGE1_PRIMAL" if cert.get("passed") else "STAGE1_PRIMAL_RESIDUAL_FAILED",
                    "passed": bool(cert.get("passed")),
                    "output_space_certificate": fallback_cert,
                },
                "fallback_used": True,
                "solution_source": "certified_stage1_primal_after_stage2_time_limit",
                "lexicographic_slack_tolerance": tol,
                "stage2_slack_budget_policy": budget_receipt,
                "residual_to_raw_target_l1_mm": float(np.sum(np.abs(d1 - t))),
                "basis": {"output_dimension": n_out, "control_dimension": n_control,
                          "hard_row_transform": hard_receipt},
            }
        return {"success": False, "status": status, "message": message,
                "nit": int(getattr(stage1, "nit", 0) + getattr(stage2, "nit", 0)),
                "objective": None, "wall_seconds": float(time.monotonic() - started),
                "lp_method": lp_method,
                "stage1": {"success": True, "objective": slack_opt},
                "stage2": {"success": False, "status": status, "message": message,
                           "time_limit": bool(time_limit)},
                "stage1_output_certificate": cert,
                "stage2_slack_budget_policy": budget_receipt,
                "fallback_used": False,
                "basis": {"output_dimension": n_out, "control_dimension": n_control,
                          "hard_row_transform": hard_receipt}}
    q = np.asarray(stage2.x[:n_control], dtype=np.float64)
    output_abs = np.asarray(stage2.x[n_control:n_control + n_out], dtype=np.float64)
    slack = np.asarray(stage2.x[n_control + n_out:], dtype=np.float64)
    d = np.asarray(M @ q).reshape(-1)
    hard_residual = np.asarray(A @ d).reshape(-1) - b
    progress = np.asarray(C @ d).reshape(-1)
    minimum_progress = float(np.min(progress)) if progress.size else 0.0
    negative_progress_rows = int(np.count_nonzero(progress < -2e-7))
    target_residual = progress + hh * slack - hh
    bound_violation = max(
        float(np.max(np.maximum(lo - d, 0.0), initial=0.0)),
        float(np.max(np.maximum(d - hi, 0.0), initial=0.0)),
        float(np.max(np.maximum(-slack, 0.0), initial=0.0)),
    )
    output_l1 = float(np.sum(output_abs))
    exact_output_l1 = float(np.sum(np.abs(d - t)))
    if (np.min(hard_residual, initial=0.0) < -2e-7
            or np.min(target_residual, initial=0.0) < -2e-7
            or bound_violation > 2e-7
            or output_l1 + 2e-7 < exact_output_l1
            or output_l1 > exact_output_l1 + 2e-6
            or float(np.dot(ww, slack)) > slack_opt + tol + 2e-7):
        raise RuntimeError("reduced-coordinate LP returned a solution outside output-space constraints/objective")
    return {
        "success": True, "status": int(stage2.status), "message": str(stage2.message),
        "nit": int(getattr(stage1, "nit", 0) + getattr(stage2, "nit", 0)),
        "objective": output_l1, "wall_seconds": float(time.monotonic() - started),
        "lp_method": lp_method, "delta": d, "basis_coordinates": q,
        "normalized_slack": slack, "normalized_slack_objective": float(np.dot(ww, slack)),
        "stage1": {"success": True, "status": int(stage1.status), "objective": slack_opt,
                   "iterations": int(getattr(stage1, "nit", 0))},
        "stage2": {"success": True, "status": int(stage2.status),
                   "objective": output_l1, "iterations": int(getattr(stage2, "nit", 0)),
                   "slack_budget": float(slack_opt + tol)},
        "lexicographic_slack_tolerance": tol,
        "stage2_slack_budget_policy": budget_receipt,
        "minimum_normalized_linear_slack": float(np.min(hard_residual, initial=np.inf)),
        "minimum_clearance_progress_mm": minimum_progress,
        "minimum_soft_target_progress_mm": minimum_progress,
        "negative_soft_target_progress_rows": negative_progress_rows,
        "soft_target_regression_allowed": True,
        "no_regression_rows_enforced": 0,
        "minimum_target_inequality_slack_mm": float(np.min(target_residual)) if target_residual.size else 0.0,
        "raw_target_l1_mm": float(np.sum(np.abs(t))),
        "residual_to_raw_target_l1_mm": exact_output_l1,
        "stage1_output_certificate": cert,
        "basis": {"output_dimension": n_out, "control_dimension": n_control,
                  "hard_row_transform": hard_receipt,
                  "output_l1_auxiliary_count": n_out},
    }

def make_solver(owner):
    """Bind this isolated copy to explicit utilities from the pinned owner."""
    wrapped=partial(_solve_sparse_l1_clearance_in_basis_without_soft_noregression,owner)
    wrapped.__name__="solve_sparse_l1_clearance_in_basis_soft_regression_470"
    wrapped.__doc__=_solve_sparse_l1_clearance_in_basis_without_soft_noregression.__doc__
    wrapped.__soft_target_no_regression_470__=True
    return wrapped

def install_solver(owner, expected_owner_sha256=EXPECTED_OWNER_SHA256):
    """Runtime-install the variant; the pinned owner source file remains unchanged."""
    path=Path(getattr(owner,"__file__","")).resolve()
    if path != EXPECTED_OWNER_PATH.resolve(): raise ValueError(f"unexpected solver owner path: {path}")
    actual=sha256_file(path)
    if actual != expected_owner_sha256: raise ValueError(f"408 owner SHA mismatch: expected {expected_owner_sha256}, got {actual}")
    if not hasattr(owner,"solve_sparse_l1_clearance_in_basis"): raise ValueError("408 owner has no pinned reduced-basis solver")
    previous=owner.solve_sparse_l1_clearance_in_basis
    replacement=make_solver(owner)
    owner.solve_sparse_l1_clearance_in_basis=replacement
    return {"variant":"soft-target-no-regression-470","owner_path":str(path),"owner_sha256":actual,
            "source_solver":"copied pinned 408 solve_sparse_l1_clearance_in_basis",
            "soft_target_noregression_rows_removed":True,"soft_target_noregression_certification_removed":True,
            "physical_A_rows_and_output_bounds_unchanged":True,"target_slack_and_weights_unchanged":True,
            "negative_soft_target_progress_reported":True,
            "previous_solver_name":getattr(previous,"__name__",type(previous).__name__),"installed_solver_name":replacement.__name__}

def install_solver(owner):
    """Install only after exact frozen-owner identity verification; return a receipt."""
    actual = Path(getattr(owner, "__file__", "")).resolve()
    if actual != EXPECTED_OWNER_PATH.resolve():
        raise ValueError(f"470 refuses unexpected owner module path: {actual}")
    digest = sha256_file(actual)
    if digest != EXPECTED_OWNER_SHA256:
        raise ValueError(f"470 refuses changed 408 owner SHA256: {digest}")
    original = owner.solve_sparse_l1_clearance_in_basis
    replacement = make_solver(owner)
    owner.solve_sparse_l1_clearance_in_basis = replacement
    return {
        "variant": "soft_target_no_regression_removed_470",
        "owner_path": str(actual),
        "owner_sha256": digest,
        "original_solver_name": getattr(original, "__name__", type(original).__name__),
        "replacement_solver_name": replacement.__name__,
        "physical_hard_rows_preserved": True,
        "output_bounds_preserved": True,
        "target_slack_inequalities_preserved": True,
        "stage1_and_stage2_no_regression_rows_enforced": 0,
        "no_regression_certificate_removed": True,
        "negative_soft_target_progress_reported": True,
        "construction_only": True,
        "native_admission": False,
    }
