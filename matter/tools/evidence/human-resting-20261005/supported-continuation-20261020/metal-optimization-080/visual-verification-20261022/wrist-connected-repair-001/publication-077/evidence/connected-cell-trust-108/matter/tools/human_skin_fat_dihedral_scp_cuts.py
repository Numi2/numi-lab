"""Advisory sequential-linearization cuts for mapped-surface dihedral guards.

Rows are local SCP cuts in displacement-from-current variables, not a global
convex certificate. Keep the exact nonlinear Float32 gate after the solve.
"""
import numpy as np
from scipy import sparse


def make_propagated_dihedral_cuts(surface, current_points, trial_points, free_nodes,
                                  edge_ids, *, limit_radians,
                                  zero_slack_tolerance=1e-10):
    """Build tangent halfspaces at a failed trial while retaining current X.

    The returned sparse matrix A and rhs b use LP displacement variables in
    millimetres relative to current_points. For each accepted edge they encode
    both signs of f(Z) + J(Z) (Y - Z) within the angle limit.

    dihedral_change_linearization returns radians/metre, so this helper applies
    the exact metre-to-millimetre factor. Any edge whose tangent cuts exclude
    zero displacement is declined. Small negative RHS values within the
    explicit numerical tolerance are clipped to zero, preserving X.
    This is an advisory local constraint-generation step only; it does not
    certify the nonlinear surface at a new point.
    """
    X = np.asarray(current_points)
    Z = np.asarray(trial_points)
    raw_free = np.asarray(free_nodes)
    raw_edges = np.asarray(edge_ids)
    if X.ndim != 2 or X.shape[1] != 3 or X.dtype.kind not in "fiu" or not np.isfinite(X).all():
        raise ValueError("current_points must be finite Nx3")
    if Z.shape != X.shape or Z.dtype.kind not in "fiu" or not np.isfinite(Z).all():
        raise ValueError("trial_points must be finite and match current_points")
    if raw_free.dtype.kind not in "iu" or raw_free.ndim != 1:
        raise ValueError("free_nodes must be an integer vector")
    F = raw_free.astype(np.int64, copy=False)
    if len(F) == 0 or len(np.unique(F)) != len(F) or np.any(F < 0) or np.any(F >= len(X)):
        raise ValueError("invalid free_nodes")
    if raw_edges.dtype.kind not in "iu" or raw_edges.ndim != 1 or len(np.unique(raw_edges)) != len(raw_edges):
        raise ValueError("edge_ids must be a unique integer vector")
    requested = raw_edges.astype(np.int64, copy=False)
    if np.any(requested < 0):
        raise ValueError("edge_ids must be nonnegative")
    limit = float(limit_radians)
    tol = float(zero_slack_tolerance)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("limit_radians must be finite and positive")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("zero_slack_tolerance must be finite and nonnegative")

    if not hasattr(surface, "metrics") or not hasattr(surface, "dihedral_passes") or not hasattr(surface, "dihedral_change_linearization"):
        raise ValueError("surface does not provide mapped-surface geometry owners")
    if not surface.dihedral_passes(X):
        raise ValueError("current_points must pass the exact configured dihedral gate")

    skipped = []
    try:
        _, _, area_ratio, normal_cosine = surface.metrics(Z)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return _empty_result(F, requested, skipped, "trial_surface_unusable:" + str(exc))
    area_ratio = np.asarray(area_ratio, dtype=np.float64)
    normal_cosine = np.asarray(normal_cosine, dtype=np.float64)
    if (area_ratio.size == 0 or normal_cosine.size == 0 or
            not np.isfinite(area_ratio).all() or not np.isfinite(normal_cosine).all()):
        return _empty_result(F, requested, skipped, "trial_surface_unusable:nonfinite_metrics")
    area_bounds = getattr(surface, "area_bounds", None)
    cosine_min = getattr(surface, "normal_cosine_min", None)
    if area_bounds is not None and (area_ratio.min() < area_bounds[0] or area_ratio.max() > area_bounds[1]):
        return _empty_result(F, requested, skipped, "trial_surface_outside_area_guard")
    if cosine_min is not None and normal_cosine.min() < cosine_min:
        return _empty_result(F, requested, skipped, "trial_surface_outside_normal_guard")

    try:
        active, fz, Jz = surface.dihedral_change_linearization(Z, F, edge_ids=requested)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return _empty_result(F, requested, skipped, "trial_dihedral_linearization_unusable:" + str(exc))
    active = np.asarray(active)
    fz = np.asarray(fz, dtype=np.float64)
    if active.dtype.kind not in "iu" or active.ndim != 1 or fz.shape != active.shape:
        raise ValueError("invalid dihedral linearization result")
    active = active.astype(np.int64, copy=False)
    if not sparse.issparse(Jz):
        raise ValueError("dihedral Jacobian must be sparse")
    Jz = Jz.tocsr().astype(np.float64)
    if Jz.shape != (len(active), 3 * len(F)):
        raise ValueError("dihedral Jacobian has wrong shape")
    Jz.sum_duplicates()
    Jz.eliminate_zeros()
    Jz.sort_indices()

    absent = np.setdiff1d(requested, active, assume_unique=False)
    skipped.extend({"edge_id": int(e), "reason": "no_free_support"} for e in absent)
    if not np.isfinite(fz).all() or not np.isfinite(Jz.data).all():
        return _empty_result(F, requested, skipped, "trial_dihedral_linearization_unusable:nonfinite")

    branch_safe = np.abs(fz) < (0.5 * np.pi)
    for row in np.flatnonzero(~branch_safe):
        skipped.append({"edge_id": int(active[row]), "reason": "wrapped_angle_branch_unsafe"})

    Jmm = (1e-3 * Jz).tocsr()
    z_mm = ((Z[F].astype(np.float64) - X[F].astype(np.float64)) * 1000.0).ravel()
    jz = np.asarray(Jmm @ z_mm).ravel()
    upper = limit - fz + jz
    lower = limit + fz - jz
    keep = np.flatnonzero(branch_safe & (np.minimum(upper, lower) >= -tol))
    for row in np.flatnonzero(branch_safe & (np.minimum(upper, lower) < -tol)):
        skipped.append({"edge_id": int(active[row]), "reason": "tangent_cut_excludes_current",
                        "upper_zero_slack_rad": float(upper[row]),
                        "lower_zero_slack_rad": float(lower[row])})

    if len(keep):
        norms = np.asarray(np.abs(Jmm[keep]).sum(axis=1)).ravel()
        redundant = norms <= np.finfo(np.float64).tiny
        for row in keep[redundant]:
            skipped.append({"edge_id": int(active[row]), "reason": "zero_jacobian"})
        keep = keep[~redundant]

    chosen = Jmm[keep].tocsr()
    upper_kept = np.maximum(upper[keep], 0.0)
    lower_kept = np.maximum(lower[keep], 0.0)
    A = sparse.vstack([chosen, -chosen], format="csr")
    b = np.r_[upper_kept, lower_kept]
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    return {
        "status": "cuts_ready" if len(keep) else "no_advisory_cuts",
        "edge_ids": active[keep].copy(),
        "jacobian_mm": chosen,
        "A_ub": A,
        "b_ub_rad": b,
        "upper_zero_slack_rad": upper_kept,
        "lower_zero_slack_rad": lower_kept,
        "trial_delta_rad": fz[keep].copy(),
        "trial_displacement_mm": z_mm.copy(),
        "skipped": skipped,
        "cut_kind": "local_scp_tangent_at_trial; exact_nonlinear_gate_required",
    }


def _empty_result(free_nodes, requested, skipped, reason):
    skipped = list(skipped)
    skipped.extend({"edge_id": int(e), "reason": reason} for e in requested)
    return {
        "status": "no_advisory_cuts",
        "edge_ids": np.empty(0, dtype=np.int64),
        "jacobian_mm": sparse.csr_matrix((0, 3 * len(free_nodes))),
        "A_ub": sparse.csr_matrix((0, 3 * len(free_nodes))),
        "b_ub_rad": np.empty(0, dtype=np.float64),
        "upper_zero_slack_rad": np.empty(0, dtype=np.float64),
        "lower_zero_slack_rad": np.empty(0, dtype=np.float64),
        "trial_delta_rad": np.empty(0, dtype=np.float64),
        "trial_displacement_mm": np.empty(3 * len(free_nodes), dtype=np.float64),
        "skipped": skipped,
        "cut_kind": "local_scp_tangent_at_trial; exact_nonlinear_gate_required",
    }

