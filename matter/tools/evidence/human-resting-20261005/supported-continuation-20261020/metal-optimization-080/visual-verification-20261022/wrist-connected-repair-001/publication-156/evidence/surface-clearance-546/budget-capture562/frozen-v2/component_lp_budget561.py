"""Opt-in sparse component decomposition for the frozen 408 reduced LP.

The stage-1 weighted slack objective is exactly separable over connected
components. Stage 2 is solved independently per component under a component
share of the same global retained-gain budget. This preserves every hard row
and gives a feasible objective-budget allocation, but can restrict global
slack redistribution; it does not claim the global stage-2 L1 optimum.
"""
from __future__ import annotations

import time
from typing import Callable
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

SCHEMA = "numi.human.component_lp_decomposition_485.v1"

def _csr(a, name):
    x = sparse.csr_matrix(a, dtype=np.float64, copy=True)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    if not np.isfinite(x.data).all():
        raise ValueError(f"{name} contains nonfinite values")
    return x

def _row_support_labels(mat: sparse.csr_matrix, control_labels: np.ndarray, name: str):
    labels = np.full(mat.shape[0], -1, dtype=np.int64)
    for r in range(mat.shape[0]):
        cols = mat.indices[mat.indptr[r]:mat.indptr[r + 1]]
        if not len(cols):
            continue
        labs = np.unique(control_labels[cols])
        if len(labs) != 1 or labs[0] < 0:
            raise ValueError(f"{name} row {r} crosses disconnected control components")
        labels[r] = int(labs[0])
    return labels

def _components_from_rows(blocks, n_control):
    K = sparse.vstack(blocks, format="csr")
    K.sum_duplicates(); K.eliminate_zeros(); K.sort_indices()
    active = np.asarray(K.getnnz(axis=0)).reshape(-1) > 0
    null_count = int(np.count_nonzero(~active))
    if not np.any(active):
        return [], np.full(n_control, -1, dtype=np.int64), null_count
    # Sparse bipartite adjacency avoids Python work proportional to every
    # nonzero; connected_components sees exact structural support only.
    nr = K.shape[0]
    zero_cc = sparse.csr_matrix((n_control, n_control), dtype=np.int8)
    zero_rr = sparse.csr_matrix((nr, nr), dtype=np.int8)
    graph = sparse.bmat([[zero_cc, K.T], [K, zero_rr]], format="csr", dtype=np.int8)
    _, graph_labels = connected_components(graph, directed=False, return_labels=True)
    raw = graph_labels[:n_control]
    groups = {}
    for col in np.flatnonzero(active):
        groups.setdefault(int(raw[col]), []).append(int(col))
    ordered = sorted((sorted(cols) for cols in groups.values()), key=lambda x: x[0])
    labels = np.full(n_control, -1, dtype=np.int64)
    for ci, cols in enumerate(ordered):
        labels[np.asarray(cols, dtype=np.int64)] = ci
    return ordered, labels, null_count

def _sparse_equal(a, b):
    delta = (a - b).tocsr()
    delta.sum_duplicates(); delta.eliminate_zeros()
    return delta.nnz == 0

def _failure(message, receipt, *, status=2, wall=0.0):
    return {"success": False, "status": int(status), "message": str(message),
            "nit": 0, "objective": None, "wall_seconds": float(wall),
            "lp_method": None, "component_decomposition_475": receipt}

def component_workload_weights(
    control_counts, hard_row_counts, clearance_row_counts,
    hard_nnz, clearance_nnz, output_map_nnz, trivial,
):
    arrays = [
        np.asarray(x, dtype=np.float64)
        for x in (
            control_counts, hard_row_counts, clearance_row_counts,
            hard_nnz, clearance_nnz, output_map_nnz,
        )
    ]
    mask = np.asarray(trivial)
    if any(x.ndim != 1 for x in arrays) or mask.ndim != 1:
        raise ValueError("component workload inputs must be one-dimensional")
    n = len(arrays[0])
    if any(len(x) != n for x in arrays) or len(mask) != n:
        raise ValueError("component workload input lengths differ")
    if mask.dtype.kind != "b":
        raise ValueError("trivial mask must contain booleans")
    if any(not np.isfinite(x).all() or np.any(x < 0.0) or np.any(x != np.floor(x)) for x in arrays):
        raise ValueError("component workload counts must be finite nonnegative integers")
    workload = np.sum(np.vstack(arrays), axis=0, dtype=np.float64)
    weights = np.where(mask, 0.0, np.maximum(workload, 1.0))
    return weights


def weighted_component_stage_limits(remaining_s, time_limit_s, workloads, active):
    remaining = float(remaining_s)
    cap = float(time_limit_s)
    weights = np.asarray(workloads, dtype=np.float64)
    active = np.asarray(active)
    if not np.isfinite(remaining) or remaining <= 0.0:
        raise ValueError("remaining wall budget must be finite and positive")
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("per-stage cap must be finite and positive")
    if weights.ndim != 1 or active.ndim != 1 or len(weights) != len(active):
        raise ValueError("workload and active arrays must be aligned vectors")
    if active.dtype.kind != "b":
        raise ValueError("active mask must contain booleans")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("component workload weights must be finite and nonnegative")
    if np.any(weights[active] <= 0.0):
        raise ValueError("every active component needs positive workload weight")
    limits = np.zeros(len(weights), dtype=np.float64)
    if not np.any(active):
        return limits
    denominator = float(np.sum(weights[active], dtype=np.float64))
    component_total_shares = remaining * weights[active] / denominator
    limits[active] = np.minimum(cap, component_total_shares / 2.0)
    if 2.0 * float(np.sum(limits, dtype=np.float64)) > remaining + 1e-9:
        raise RuntimeError("weighted two-stage allocations exceed remaining wall budget")
    return limits


def solve_by_components(original_solver: Callable, A, b, target_output, lower_output, upper_output,
                        C, h, weights, output_map, time_limit_s, slack_tolerance=1e-9,
                        allow_stage1_feasible_timeout_fallback=False, lp_method="highs",
                        retained_gain_fraction=None, *, allow_soft_target_regression=False,
                        max_total_wall_s=120.0):
    """Call frozen solver independently on exact sparse components.

    Soft target regression is an opt-in for an already-installed 470 solver;
    it removes only the C@delta >= 0 certificate. Target slack, hard rows,
    output bounds, and global slack budget remain certified.
    """
    if type(allow_soft_target_regression) is not bool:
        raise ValueError("allow_soft_target_regression must be a bool")
    if allow_soft_target_regression and not getattr(original_solver, "__soft_target_no_regression_470__", False):
        raise ValueError("485 soft-target regression requires the explicit 470 solver marker")
    started = time.monotonic()
    A = _csr(A, "A"); C = _csr(C, "C"); M = _csr(output_map, "output_map")
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    t = np.asarray(target_output, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower_output, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper_output, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n_out, n_control = M.shape; m = len(h)
    if A.shape != (len(b), n_out) or C.shape != (m, n_out) or t.shape != (n_out,) or lo.shape != (n_out,) or hi.shape != (n_out,) or w.shape != (m,):
        raise ValueError("component LP dimensions do not align")
    if not all(np.isfinite(x).all() for x in (b, t, lo, hi, h, w)):
        raise ValueError("component LP vectors contain nonfinite values")
    if np.any(lo > hi) or np.any(h <= 0) or np.any(w < 0) or (m and float(w.sum()) <= 0):
        raise ValueError("component LP has invalid bounds, target distances, or weights")
    if n_control == 0 or max_total_wall_s <= 0 or time_limit_s <= 0:
        raise ValueError("component LP needs controls and positive time bounds")

    AM = _csr(A @ M, "A@M")
    CM = _csr(C @ M, "C@M")
    groups, control_labels, null_control_count = _components_from_rows((AM, CM, M), n_control)
    a_labels = _row_support_labels(AM, control_labels, "A@M")
    c_labels = _row_support_labels(CM, control_labels, "C@M")
    m_labels = _row_support_labels(M, control_labels, "M")
    output_rows_by_component = [np.flatnonzero(m_labels == i) for i in range(len(groups))]
    a_rows_by_component = [np.flatnonzero(a_labels == i) for i in range(len(groups))]
    c_rows_by_component = [np.flatnonzero(c_labels == i) for i in range(len(groups))]
    zero_a = np.flatnonzero(a_labels < 0)
    zero_c = np.flatnonzero(c_labels < 0)
    zero_out = np.flatnonzero(m_labels < 0)
    receipt = {
        "schema": SCHEMA, "classification": "construction-only sparse LP decomposition; exact hard rows retained",
        "algorithm": "connected components of row-control bipartite graph for vstack(A@M,C@M,M)",
        "component_count": int(len(groups)), "component_control_counts": [int(len(x)) for x in groups],
        "component_output_counts": [int(len(x)) for x in output_rows_by_component],
        "component_hard_row_counts": [int(len(x)) for x in a_rows_by_component],
        "component_clearance_row_counts": [int(len(x)) for x in c_rows_by_component],
        "hard_rows_fixed_zero_count": int(len(zero_a)), "clearance_rows_fixed_zero_count": int(len(zero_c)),
        "output_rows_fixed_zero_count": int(len(zero_out)), "null_control_count": null_control_count,
        "original_hard_rows": int(A.shape[0]), "original_clearance_rows": int(C.shape[0]),
        "original_output_rows": int(n_out), "original_control_count": int(n_control),
        "max_total_wall_s": float(max_total_wall_s),
        "stage2_allocation_policy": "per-component share of same retained-gain fraction; exact summed budget, but componentwise allocation can restrict global L1 optimum",
        "global_stage2_l1_optimality_claim": False, "native_admission": False,
        "allow_soft_target_regression": bool(allow_soft_target_regression),
        "soft_target_no_regression_rows_enforced": not bool(allow_soft_target_regression),
    }
    zero_feasible = bool(not np.any(b > 0.0) and not np.any(lo > 0.0) and not np.any(hi < 0.0))
    if np.any(b[zero_a] > 0.0):
        receipt["constant_hard_row_failure_indices"] = np.flatnonzero(b[zero_a] > 0.0).astype(int).tolist()
        return _failure("a hard row with zero reduced coefficients has positive lower bound", receipt, wall=time.monotonic()-started)
    if np.any(lo[zero_out] > 0.0) or np.any(hi[zero_out] < 0.0):
        return _failure("a fixed-zero output row violates its output bound", receipt, wall=time.monotonic()-started)

    # Exact cancellation can make a nonzero source row identically zero after M.
    # Certify that the reduced local functional reconstructs each assigned row.
    for ci, cols_list in enumerate(groups):
        cols = np.asarray(cols_list, dtype=np.int64)
        outs = output_rows_by_component[ci]
        ar = a_rows_by_component[ci]
        cr = c_rows_by_component[ci]
        if len(ar) and not _sparse_equal(AM[ar][:, cols], A[ar][:, outs] @ M[outs][:, cols]):
            raise ValueError(f"hard-row projection mismatch in component {ci}")
        if len(cr) and not _sparse_equal(CM[cr][:, cols], C[cr][:, outs] @ M[outs][:, cols]):
            raise ValueError(f"clearance-row projection mismatch in component {ci}")

    delta = np.zeros(n_out, dtype=np.float64)
    q = np.zeros(n_control, dtype=np.float64)
    slack = np.ones(m, dtype=np.float64)
    component_reports = []
    stage1_sum = float(np.sum(w[zero_c], dtype=np.float64))
    nit_sum = 0
    fallback_components = []
    component_count_with_clearance = sum(bool(len(x)) for x in c_rows_by_component)
    component_count_for_tolerance = max(1, component_count_with_clearance)
    trivial=[]
    for ci in range(len(groups)):
        outs=output_rows_by_component[ci];ar=a_rows_by_component[ci];cr=c_rows_by_component[ci]
        trivial.append(not len(cr) and np.all(t[outs]==0) and np.all(b[ar]<=0) and np.all(lo[outs]<=0) and np.all(hi[outs]>=0))
    receipt["analytic_zero_component_count"]=int(sum(trivial))
    component_workload_parts = []
    for ci in range(len(groups)):
        outs = output_rows_by_component[ci]
        ar = a_rows_by_component[ci]
        cr = c_rows_by_component[ci]
        component_workload_parts.append({
            "component": int(ci),
            "controls": int(len(groups[ci])),
            "hard_rows": int(len(ar)),
            "clearance_rows": int(len(cr)),
            "hard_nnz": int(AM[ar].nnz),
            "clearance_nnz": int(CM[cr].nnz),
            "output_map_nnz": int(M[outs].nnz),
            "analytic_zero": bool(trivial[ci]),
        })
    workloads = component_workload_weights(
        [x["controls"] for x in component_workload_parts],
        [x["hard_rows"] for x in component_workload_parts],
        [x["clearance_rows"] for x in component_workload_parts],
        [x["hard_nnz"] for x in component_workload_parts],
        [x["clearance_nnz"] for x in component_workload_parts],
        [x["output_map_nnz"] for x in component_workload_parts],
        trivial,
    )
    for part, workload in zip(component_workload_parts, workloads):
        part["workload_weight"] = float(workload)
    receipt["component_workload_parts"] = component_workload_parts
    receipt["component_workload_weights"] = workloads.tolist()
    receipt["wall_budget_allocation_policy"] = (
        "same max_total_wall_s budget allocated dynamically by remaining workload weight: "
        "controls + hard rows + clearance rows + nnz(A@M) + nnz(C@M) + nnz(M); "
        "two LP stages share each component allotment, per-stage cap remains time_limit_s, "
        "analytic-zero components are excluded"
    )
    receipt["component_time_limit_per_stage_s"] = [0.0 if x else None for x in trivial]
    receipt["active_stage_time_allocation"] = True
    for ci, cols_list in enumerate(groups):
        elapsed = time.monotonic() - started
        remaining = float(max_total_wall_s) - elapsed
        if remaining <= 0.05:
            receipt["component_results"] = component_reports
            return _failure("total component LP time budget exhausted before all components solved", receipt, status=1, wall=elapsed)
        cols = np.asarray(cols_list, dtype=np.int64)
        outs = output_rows_by_component[ci]
        ar = a_rows_by_component[ci]
        cr = c_rows_by_component[ci]
        Mloc = M[outs][:, cols].tocsr()
        Aloc = A[ar][:, outs].tocsr()
        Cloc = C[cr][:, outs].tocsr()
        # Re-share current remaining wall budget by workload among nontrivial
        # components not yet solved. Each component's two LP stages share its
        # allotment; the original per-stage cap and global wall limit are unchanged.
        remaining_active = ~np.asarray(trivial[ci:], dtype=bool)
        remaining_limits = weighted_component_stage_limits(
            remaining,
            float(time_limit_s),
            workloads[ci:],
            remaining_active,
        )
        per_stage_limit = float(remaining_limits[0])
        if not trivial[ci] and per_stage_limit <= 0.0:
            receipt["component_results"] = component_reports
            return _failure("weighted component budget assigned no positive time", receipt, status=1, wall=elapsed)
        receipt["component_time_limit_per_stage_s"][ci] = per_stage_limit
        if retained_gain_fraction is not None and float(retained_gain_fraction) < 1.0 and not zero_feasible:
            local_fraction = 1.0
            local_tolerance = 0.0
        elif retained_gain_fraction is None or float(retained_gain_fraction) == 1.0:
            local_fraction = retained_gain_fraction
            local_tolerance = float(slack_tolerance) / component_count_for_tolerance
        else:
            local_fraction = float(retained_gain_fraction)
            local_tolerance = float(slack_tolerance)
        if trivial[ci]:
            solved=dict(success=True,status=0,message="Analytic zero: no target rows, zero L1 target and independently feasible zero displacement",nit=0,delta=np.zeros(len(outs)),basis_coordinates=np.zeros(len(cols)),normalized_slack=np.empty(0),stage1={"objective":0.},wall_seconds=0.,fallback_used=False)
        else:
            solved = original_solver(
                Aloc, b[ar], t[outs], lo[outs], hi[outs], Cloc, h[cr], w[cr], Mloc,
                per_stage_limit, slack_tolerance=local_tolerance,
                allow_stage1_feasible_timeout_fallback=allow_stage1_feasible_timeout_fallback,
                lp_method=lp_method, retained_gain_fraction=local_fraction)
        nit_sum += int(solved.get("nit", 0))
        component_receipt = {
            "component": ci, "control_rows": int(len(cols)), "output_rows": int(len(outs)),
            "hard_rows": int(len(ar)), "clearance_rows": int(len(cr)),
            "time_limit_per_stage_s": per_stage_limit,
            "workload_weight": float(workloads[ci]),
            "success": bool(solved.get("success")),
            "status": solved.get("status"), "message": solved.get("message"),
            "stage1_objective": (solved.get("stage1") or {}).get("objective"),
            "stage2_slack_budget_policy": solved.get("stage2_slack_budget_policy"),
            "fallback_used": bool(solved.get("fallback_used", False)),
            "wall_seconds": solved.get("wall_seconds"),
        }
        component_reports.append(component_receipt)
        if not solved.get("success"):
            receipt["component_results"] = component_reports
            return _failure(f"component {ci} LP failed: {solved.get('message')}", receipt, status=int(solved.get("status", 2)), wall=time.monotonic()-started)
        local_delta = np.asarray(solved.get("delta"), dtype=np.float64).reshape(-1)
        local_q = np.asarray(solved.get("basis_coordinates"), dtype=np.float64).reshape(-1)
        local_slack = np.asarray(solved.get("normalized_slack"), dtype=np.float64).reshape(-1)
        if local_delta.shape != (len(outs),) or local_q.shape != (len(cols),) or local_slack.shape != (len(cr),) or not np.isfinite(local_delta).all() or not np.isfinite(local_q).all() or not np.isfinite(local_slack).all():
            receipt["component_results"] = component_reports
            return _failure(f"component {ci} returned malformed solution arrays", receipt, wall=time.monotonic()-started)
        delta[outs] = local_delta
        q[cols] = local_q
        slack[cr] = local_slack
        stage1_sum += float((solved.get("stage1") or {}).get("objective", solved.get("normalized_slack_objective", 0.0)))
        if solved.get("fallback_used"):
            fallback_components.append(ci)

    mapped_delta = np.asarray(M @ q, dtype=np.float64).reshape(-1)
    if not np.allclose(mapped_delta, delta, rtol=0.0, atol=2e-9):
        receipt["component_results"] = component_reports
        receipt["global_output_map_max_abs_error"] = float(np.max(np.abs(mapped_delta-delta), initial=0.0))
        return _failure("component outputs do not reconstruct through the original global output map", receipt, wall=time.monotonic()-started)
    delta = mapped_delta
    hard_resid = np.asarray(A @ delta).reshape(-1) - b
    progress = np.asarray(C @ delta).reshape(-1)
    target_resid = progress + h * slack - h
    bound_error = max(float(np.max(np.maximum(lo-delta,0.0), initial=0.0)),
                      float(np.max(np.maximum(delta-hi,0.0), initial=0.0)),
                      float(np.max(np.maximum(-slack,0.0), initial=0.0)))
    weighted = float(np.dot(w, slack))
    global_stage2_budget = None
    if retained_gain_fraction is None or float(retained_gain_fraction) == 1.0:
        global_stage2_budget = float(stage1_sum + slack_tolerance)
    elif zero_feasible:
        global_stage2_budget = float(stage1_sum + (1.0-float(retained_gain_fraction))*max(0.0,float(w.sum())-stage1_sum))
    else:
        global_stage2_budget = float(stage1_sum)
    minimum_progress = float(np.min(progress, initial=np.inf))
    negative_progress_rows = int(np.count_nonzero(progress < -2e-7))
    progress_pass = bool(allow_soft_target_regression or minimum_progress >= -2e-7)
    cert_pass = bool(np.min(hard_resid, initial=np.inf) >= -2e-7 and
                     progress_pass and
                     np.min(target_resid, initial=np.inf) >= -2e-7 and
                     bound_error <= 2e-7 and weighted <= global_stage2_budget + 2e-7)
    if not cert_pass:
        receipt.update({"component_results": component_reports, "global_stage2_budget": global_stage2_budget,
                        "weighted_slack": weighted, "minimum_hard_residual": float(np.min(hard_resid, initial=np.inf)),
                        "minimum_soft_target_progress_mm": minimum_progress,
                        "negative_soft_target_progress_rows": negative_progress_rows,
                        "soft_target_regression_allowed": bool(allow_soft_target_regression),
                        "minimum_target_residual": float(np.min(target_resid, initial=np.inf)),
                        "maximum_bound_violation": bound_error})
        return _failure("decomposed solution failed full original-space certification", receipt, wall=time.monotonic()-started)
    receipt.update({
        "component_results": component_reports, "global_zero_move_baseline_feasible": zero_feasible,
        "stage1_objective_sum": stage1_sum, "final_weighted_slack_recomputed": weighted,
        "global_stage2_slack_budget": global_stage2_budget, "global_weighted_slack": weighted,
        "global_stage2_budget_pass": bool(weighted <= global_stage2_budget + 2e-7),
        "fallback_component_indices": fallback_components,
        "minimum_hard_residual": float(np.min(hard_resid, initial=np.inf)),
        "minimum_no_regression_residual": (None if allow_soft_target_regression else minimum_progress),
        "minimum_soft_target_progress_mm": minimum_progress,
        "negative_soft_target_progress_rows": negative_progress_rows,
        "soft_target_regression_allowed": bool(allow_soft_target_regression),
        "soft_target_no_regression_rows_enforced": not bool(allow_soft_target_regression),
        "minimum_target_residual": float(np.min(target_resid, initial=np.inf)),
        "maximum_output_bound_violation": bound_error,
        "output_l1_mm": float(np.sum(np.abs(delta-t))),
        "global_original_space_certificate_pass": True,
        "global_stage2_l1_optimality_claim": False,
        "wall_seconds": float(time.monotonic()-started),
    })
    return {
        "success": True, "status": 0, "message": "all sparse components solved and global original-space certificate passed",
        "nit": nit_sum, "objective": float(np.sum(np.abs(delta-t))),
        "wall_seconds": float(time.monotonic()-started), "lp_method": lp_method,
        "fallback_used": bool(fallback_components),
        "delta": delta, "basis_coordinates": q, "normalized_slack": slack,
        "normalized_slack_objective": weighted,
        "stage1": {"success": True, "status": 0, "objective": stage1_sum, "component_count": len(groups)},
        "stage2": {"success": not bool(fallback_components),
                    "status": (1 if fallback_components else 0),
                    "message": ("one or more components used a certified stage-1 primal after stage-2 timeout; stage-2 tie-break incomplete" if fallback_components else "all component stage-2 solves completed"),
                    "objective": (None if fallback_components else float(np.sum(np.abs(delta-t)))),
                    "slack_budget": global_stage2_budget, "optimality_claim": False,
                    "fallback_components": fallback_components},
        "stage2_slack_budget_policy": {"global_budget": global_stage2_budget,
                    "componentwise_distribution": "same retained fraction per component; sum equals global budget when zero baseline feasible",
                    "global_stage2_l1_optimality_claim": False},
        "minimum_normalized_linear_slack": float(np.min(hard_resid, initial=np.inf)),
        "minimum_clearance_progress_mm": minimum_progress,
        "minimum_soft_target_progress_mm": minimum_progress,
        "negative_soft_target_progress_rows": negative_progress_rows,
        "soft_target_regression_allowed": bool(allow_soft_target_regression),
        "soft_target_no_regression_rows_enforced": not bool(allow_soft_target_regression),
        "minimum_target_inequality_slack_mm": float(np.min(target_resid, initial=np.inf)),
        "raw_target_l1_mm": float(np.sum(np.abs(t))),
        "residual_to_raw_target_l1_mm": float(np.sum(np.abs(delta-t))),
        "basis": {"output_dimension": int(n_out), "control_dimension": int(n_control),
                  "component_count": int(len(groups)), "output_l1_auxiliary_count": int(n_out)},
        "component_decomposition_475": receipt,
    }

def install(owner, *, allow_soft_target_regression=False, max_total_wall_s=120.0):
    """Install component solver; soft regression requires prior explicit 470 install."""
    original = owner.solve_sparse_l1_clearance_in_basis
    if type(allow_soft_target_regression) is not bool:
        raise ValueError("allow_soft_target_regression must be a bool")
    marker = bool(getattr(original, "__soft_target_no_regression_470__", False))
    if allow_soft_target_regression and not marker:
        raise ValueError("485 refuses soft regression without the 470 solver marker")
    if getattr(owner, "_component_lp_decomposition_485_installed", False):
        raise RuntimeError("component solver already installed")
    def wrapped(A,b,target,lower,upper,C,h,weights,output_map,time_limit_s,slack_tolerance=1e-9,
                allow_stage1_feasible_timeout_fallback=False,lp_method="highs",retained_gain_fraction=None):
        return solve_by_components(original,A,b,target,lower,upper,C,h,weights,output_map,time_limit_s,
                 slack_tolerance,allow_stage1_feasible_timeout_fallback,lp_method,retained_gain_fraction,
                 allow_soft_target_regression=allow_soft_target_regression,
                 max_total_wall_s=max_total_wall_s)
    owner.solve_sparse_l1_clearance_in_basis = wrapped
    owner._component_lp_decomposition_485_installed = True
    owner._component_lp_decomposition_485_receipt = {
        "schema": SCHEMA, "max_total_wall_s": float(max_total_wall_s),
        "allow_soft_target_regression": bool(allow_soft_target_regression),
        "soft_target_no_regression_470_marker_verified": marker,
        "hard_rows_and_bounds_unchanged": True,
        "target_slack_and_global_budget_unchanged": True,
        "native_admission": False,
    }
    return owner._component_lp_decomposition_485_receipt
