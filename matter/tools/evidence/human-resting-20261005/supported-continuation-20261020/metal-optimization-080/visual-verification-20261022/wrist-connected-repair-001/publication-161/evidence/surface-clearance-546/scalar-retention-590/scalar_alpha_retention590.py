"""Bounded scalar-alpha LP proposal math for saved coordinate fields.

This module creates construction-only controls. The frozen 582 owner remains
responsible for Float32 area/source/trust/self checks and witness halving.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

@dataclass(frozen=True)
class AreaCut590:
    face_row: int
    kind: str
    variable_rows: np.ndarray
    coefficients: np.ndarray
    rhs: float
    value_at_center: float


def scalarize_edge_rows590(edge_rows, delta_mm) -> sparse.csr_matrix:
    """Convert E_xyz @ (alpha_i * delta_i) into equivalent sparse E_alpha."""
    E = sparse.csr_matrix(edge_rows, dtype=np.float64)
    d = np.asarray(delta_mm, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 3 or E.shape[1] != 3 * len(d):
        raise ValueError("edge rows and node displacement domains differ")
    if not np.isfinite(E.data).all() or not np.isfinite(d).all():
        raise ValueError("edge rows and displacement must be finite")
    row_ids = np.repeat(np.arange(E.shape[0], dtype=np.int64), np.diff(E.indptr))
    scalar_cols = E.indices.astype(np.int64, copy=False) // 3
    coeff = E.data * d.reshape(-1)[E.indices]
    A = sparse.coo_matrix((coeff, (row_ids, scalar_cols)), shape=(E.shape[0], len(d)))
    A.sum_duplicates()
    return A.tocsr()


def analytic_alpha_caps590(base_mm, source_mm, delta_mm, *, trust_mm: float,
                            source_cap_mm: float, previous_upper=None) -> np.ndarray:
    """Per-node exact segment caps for trust and source-centered spheres."""
    base = np.asarray(base_mm, dtype=np.float64)
    source = np.asarray(source_mm, dtype=np.float64)
    delta = np.asarray(delta_mm, dtype=np.float64)
    if (base.ndim != 2 or base.shape[1] != 3 or source.shape != base.shape
            or delta.shape != base.shape or not all(np.isfinite(v).all() for v in (base, source, delta))):
        raise ValueError("base/source/delta must be finite aligned Nx3 arrays in millimetres")
    if (not math.isfinite(trust_mm) or trust_mm <= 0
            or not math.isfinite(source_cap_mm) or source_cap_mm <= 0):
        raise ValueError("trust and source caps must be finite and positive")
    if previous_upper is None:
        upper = np.ones(len(base), dtype=np.float64)
    else:
        upper = np.asarray(previous_upper, dtype=np.float64).copy()
        if upper.shape != (len(base),) or not np.isfinite(upper).all() or np.any(upper < 0) or np.any(upper > 1):
            raise ValueError("previous alpha bounds must be a finite [0,1] node vector")
    d2 = np.einsum("ij,ij->i", delta, delta)
    dnorm = np.sqrt(d2)
    moving = dnorm > 0
    upper[moving] = np.minimum(upper[moving], trust_mm / dnorm[moving])
    v = base - source
    v2 = np.einsum("ij,ij->i", v, v)
    cap2 = float(source_cap_mm) ** 2
    tol2 = max(1e-10, 2.0 * source_cap_mm * 1e-6)
    if np.any(v2 > cap2 + tol2):
        bad = np.flatnonzero(v2 > cap2 + tol2)
        raise ValueError(f"base violates source sphere at rows {bad[:16].tolist()}")
    b = np.einsum("ij,ij->i", v, delta)
    slack = np.maximum(0.0, cap2 - v2)
    disc = np.maximum(0.0, b * b + d2 * slack)
    root_disc = np.sqrt(disc)
    cap = np.ones(len(base), dtype=np.float64)
    for i in np.flatnonzero(moving):
        if b[i] >= 0:
            denom = root_disc[i] + b[i]
            cap[i] = slack[i] / denom if denom > 0 else 0.0
        else:
            cap[i] = (-b[i] + root_disc[i]) / d2[i]
        cap[i] = max(0.0, cap[i])
    return np.clip(np.minimum(upper, cap), 0.0, 1.0)


def face_area_values_gradients590(base_mm, reference_mm, faces, free_rows,
                                  delta_mm, alpha):
    """Current double-area ratios and derivatives with respect to node alphas."""
    base = np.asarray(base_mm, dtype=np.float64)
    ref = np.asarray(reference_mm, dtype=np.float64)
    F = np.asarray(faces)
    free = np.asarray(free_rows)
    d = np.asarray(delta_mm, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    if (base.ndim != 2 or base.shape[1] != 3 or ref.shape != base.shape
            or F.ndim != 2 or F.shape[1] != 3 or F.dtype.kind not in "iu"
            or np.any(F < 0) or np.any(F >= len(base))):
        raise ValueError("invalid area geometry domains")
    if free.ndim != 1 or free.dtype.kind not in "iu" or d.shape != (len(free), 3) or a.shape != (len(free),):
        raise ValueError("invalid scalar control domain")
    if not all(np.isfinite(v).all() for v in (base, ref, d, a)):
        raise ValueError("area geometry and controls must be finite")
    if len(np.unique(free)) != len(free) or np.any(free < 0) or np.any(free >= len(base)):
        raise ValueError("free node rows must be unique and in domain")
    if np.any(a < 0) or np.any(a > 1):
        raise ValueError("alpha values must be in [0,1]")
    idx = np.full(len(base), -1, dtype=np.int64)
    idx[free.astype(np.int64)] = np.arange(len(free), dtype=np.int64)
    directions = np.zeros_like(base)
    directions[free.astype(np.int64)] = d
    Y = base.copy()
    Y[free.astype(np.int64)] += a[:, None] * d
    p, q = Y[F[:, 1]] - Y[F[:, 0]], Y[F[:, 2]] - Y[F[:, 0]]
    c = np.cross(p, q)
    bp, bq = base[F[:, 1]] - base[F[:, 0]], base[F[:, 2]] - base[F[:, 0]]
    cbase = np.cross(bp, bq)
    rp, rq = ref[F[:, 1]] - ref[F[:, 0]], ref[F[:, 2]] - ref[F[:, 0]]
    cref = np.cross(rp, rq)
    base_n2 = np.einsum("ij,ij->i", cbase, cbase)
    ref_n = np.linalg.norm(cref, axis=1)
    cur_n = np.linalg.norm(c, axis=1)
    if np.any(base_n2 <= 0) or np.any(ref_n <= 0):
        raise ValueError("degenerate base or fixed-reference face")
    oriented = np.einsum("ij,ij->i", c, cbase) / base_n2
    magnitude = cur_n / ref_n
    d0, d1, d2 = directions[F[:, 0]], directions[F[:, 1]], directions[F[:, 2]]
    dc0 = np.cross(-d0, q) + np.cross(p, -d0)
    dc1 = np.cross(d1, q)
    dc2 = np.cross(p, d2)
    dcs = (dc0, dc1, dc2)
    grad_oriented = np.stack([np.einsum("ij,ij->i", dc, cbase) / base_n2 for dc in dcs], axis=1)
    grad_magnitude = np.zeros_like(grad_oriented)
    good = cur_n > np.finfo(np.float64).tiny
    for corner, dc in enumerate(dcs):
        grad_magnitude[good, corner] = np.einsum("ij,ij->i", c[good], dc[good]) / (cur_n[good] * ref_n[good])
    return oriented, magnitude, grad_oriented, grad_magnitude


def build_area_cuts590(base_mm, reference_mm, faces, free_rows, delta_mm, alpha,
                       *, floor: float = 0.1, cut_margin: float = 1e-5,
                       near_band: float = 0.02):
    """Build local tangent cuts near the unchanged 0.1 area/orientation gates."""
    if not (math.isfinite(floor) and 0 < floor < 1 and math.isfinite(cut_margin) and cut_margin >= 0
            and math.isfinite(near_band) and near_band >= 0):
        raise ValueError("invalid area cut settings")
    F = np.asarray(faces, dtype=np.int64)
    free = np.asarray(free_rows, dtype=np.int64)
    a = np.asarray(alpha, dtype=np.float64)
    orient, mag, go, gm = face_area_values_gradients590(base_mm, reference_mm, F, free, delta_mm, a)
    orient0, mag0, _, _ = face_area_values_gradients590(base_mm, reference_mm, F, free, delta_mm, np.zeros(len(free), dtype=np.float64))
    local = np.full(len(np.asarray(base_mm)), -1, dtype=np.int64)
    local[free] = np.arange(len(free), dtype=np.int64)
    active, blocked = {}, []
    threshold = floor + near_band
    for fi, face in enumerate(F):
        for kind, values, gradients in (("current_oriented", orient, go),
                                        ("fixed_reference_magnitude", mag, gm)):
            value = float(values[fi])
            if value >= threshold and value >= floor:
                continue
            locs = local[face]
            keep = locs >= 0
            vars_ = locs[keep]
            coeff = np.asarray(gradients[fi, keep], dtype=np.float64)
            base_value = float(orient0[fi] if kind == "current_oriented" else mag0[fi])
            target_floor = min(float(floor + cut_margin), base_value)
            cut_rhs = float(target_floor - value + np.dot(coeff, a[vars_]))
            if len(vars_) == 0 or not np.any(coeff):
                if value < floor + cut_margin:
                    blocked.append({"face_row": int(fi), "kind": kind, "value": value})
                continue
            active[(int(fi), kind)] = AreaCut590(int(fi), kind, vars_.copy(), coeff.copy(), cut_rhs, value)
    return active, blocked, {
        "minimum_current_oriented_ratio": float(orient.min(initial=np.inf)),
        "minimum_fixed_reference_magnitude_ratio": float(mag.min(initial=np.inf)),
        "near_face_count": int(np.count_nonzero(np.minimum(orient, mag) < threshold)),
        "area_failure_faces": np.flatnonzero((orient < floor) | (mag < floor)).astype(int).tolist(),
    }


def pack_area_cuts590(cuts, nvar: int):
    ordered = [cuts[k] for k in sorted(cuts)]
    rows, cols, vals, rhs = [], [], [], []
    for i, cut in enumerate(ordered):
        rows.extend([i] * len(cut.variable_rows))
        cols.extend(cut.variable_rows.tolist())
        vals.extend(cut.coefficients.tolist())
        rhs.append(cut.rhs)
    A = sparse.coo_matrix((np.asarray(vals, dtype=np.float64),
                           (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
                          shape=(len(ordered), nvar)).tocsr()
    return A, np.asarray(rhs, dtype=np.float64), ordered


def certify_alpha590(alpha, upper, edge_matrix, edge_rhs, area_matrix, area_rhs, *, tolerance=1e-7):
    """Check one LP vector against every packed bound and scalarized row."""
    a=np.asarray(alpha,dtype=np.float64);u=np.asarray(upper,dtype=np.float64)
    E=sparse.csr_matrix(edge_matrix,dtype=np.float64);er=np.asarray(edge_rhs,dtype=np.float64)
    A=sparse.csr_matrix(area_matrix,dtype=np.float64);ar=np.asarray(area_rhs,dtype=np.float64)
    if a.ndim!=1 or u.shape!=a.shape or E.shape[1]!=len(a) or A.shape[1]!=len(a):
        raise ValueError("LP certificate domains differ")
    if er.shape!=(E.shape[0],) or ar.shape!=(A.shape[0],) or not np.isfinite(a).all() or not np.isfinite(u).all():
        raise ValueError("LP certificate rows or values invalid")
    edge_slack=E@a-er;area_slack=A@a-ar
    lower_violation=float(np.maximum(-a,0).max(initial=0.0))
    upper_violation=float(np.maximum(a-u,0).max(initial=0.0))
    min_edge=float(edge_slack.min(initial=np.inf));min_area=float(area_slack.min(initial=np.inf))
    passed=(lower_violation<=tolerance and upper_violation<=tolerance and min_edge>=-tolerance and min_area>=-tolerance)
    return {"pass":bool(passed),"lower_bound_violation":lower_violation,"upper_bound_violation":upper_violation,
            "minimum_edge_row_slack":min_edge,"minimum_area_cut_slack":min_area,"tolerance":float(tolerance)}


def optimize_scalar_alpha590(edge_rows, edge_rhs, base_mm, source_mm, reference_mm,
                              faces, free_rows, delta_mm, *, trust_mm=4.0,
                              source_cap_mm=12.0, edge_gradient=0.5,
                              area_floor=0.1, cut_margin=1e-5, near_band=0.02,
                              max_lp_calls=8, lp_time_limit_s=5.0):
    """Maximize sum(alpha_i) under scalarized 408 edge rows and local area cuts."""
    E = sparse.csr_matrix(edge_rows, dtype=np.float64)
    rhs = np.asarray(edge_rhs, dtype=np.float64)
    free = np.asarray(free_rows)
    d = np.asarray(delta_mm, dtype=np.float64)
    base = np.asarray(base_mm, dtype=np.float64)
    src = np.asarray(source_mm, dtype=np.float64)
    if free.ndim != 1 or free.dtype.kind not in "iu" or len(free) == 0:
        raise ValueError("scalar retention needs a nonempty integer free-node scope")
    if len(np.unique(free)) != len(free) or np.any(free < 0) or np.any(free >= len(base)):
        raise ValueError("free node rows are invalid")
    if d.shape != (len(free), 3) or E.shape[1] != 3 * len(free) or rhs.shape != (E.shape[0],):
        raise ValueError("edge and scalar-variable domains differ")
    if not (1 <= int(max_lp_calls) <= 8 and math.isfinite(lp_time_limit_s) and 0 < lp_time_limit_s <= 5.0):
        raise ValueError("LP budget must be at most eight calls of at most five seconds")
    Aedge = scalarize_edge_rows590(E, d)
    upper = analytic_alpha_caps590(base[free], src[free], d, trust_mm=trust_mm,
                                    source_cap_mm=source_cap_mm)
    if not np.isfinite(upper).all():
        raise ValueError("analytic alpha bounds are nonfinite")
    cuts, history = {}, []
    alpha = np.zeros(len(free), dtype=np.float64)
    for call in range(int(max_lp_calls)):
        Aarea, barea, ordered = pack_area_cuts590(cuts, len(free))
        Aineq = sparse.vstack((-Aedge, -Aarea), format="csr") if len(barea) else -Aedge
        bineq = np.concatenate((-rhs, -barea)) if len(barea) else -rhs
        res = linprog(-np.ones(len(free), dtype=np.float64), A_ub=Aineq, b_ub=bineq,
                      bounds=[(0.0, float(v)) for v in upper], method="highs",
                      options={"time_limit": float(lp_time_limit_s)})
        rec = {"call": call, "solver_status": int(res.status), "solver_success": bool(res.success),
               "solver_message": str(res.message), "area_cut_count": int(len(ordered)),
               "edge_row_count": int(E.shape[0]), "time_limit_s": float(lp_time_limit_s)}
        if res.x is None or np.asarray(res.x).shape != (len(free),) or not np.isfinite(res.x).all():
            history.append(rec)
            rec["incumbent_certificate"] = None
            if "last_feasible_alpha" not in locals():
                alpha=np.zeros(len(free),dtype=np.float64)
            else:
                alpha=last_feasible_alpha.copy()
                rec["fallback"] = "retained previous certified scalar proposal"
            break
        trial_alpha = np.asarray(res.x, dtype=np.float64)
        certificate=certify_alpha590(trial_alpha,upper,Aedge,rhs,Aarea,barea)
        rec["incumbent_certificate"]=certificate
        rec["alpha_sum"] = float(trial_alpha.sum())
        if not certificate["pass"]:
            rec["feasible_incumbent_rejected"] = "bounds or packed scalar edge/area rows"
            history.append(rec)
            if "last_feasible_alpha" not in locals():
                alpha=np.zeros(len(free),dtype=np.float64)
            else:
                alpha=last_feasible_alpha.copy()
                rec["fallback"] = "retained previous certified scalar proposal"
            break
        alpha = np.clip(trial_alpha, 0.0, upper)
        last_feasible_alpha=alpha.copy()
        rec["feasible_incumbent_only"] = bool(not res.success)
        new_cuts, blocked, diag = build_area_cuts590(
            base, reference_mm, faces, free, d, alpha, floor=area_floor,
            cut_margin=cut_margin, near_band=near_band)
        rec.update(diag)
        rec["blocked_area_cut_count"] = len(blocked)
        rec["bounded_cut_keys"] = [f"{k[0]}:{k[1]}" for k in sorted(new_cuts)]
        history.append(rec)
        if diag["area_failure_faces"] and not new_cuts:
            break
        if diag["near_face_count"] == 0 or not new_cuts:
            break
        # Sequential tangent cuts and monotone upper bounds give bounded local
        # retention. Frozen 582 still checks the actual Float32 candidate.
        upper = np.minimum(upper, alpha)
        cuts = new_cuts
        if np.all(upper <= 1e-14):
            break
    return alpha, {
        "lp_calls": len(history), "lp_history": history,
        "alpha_sum": float(alpha.sum()),
        "alpha_min": float(alpha.min()) if len(alpha) else 0.0,
        "alpha_max": float(alpha.max()) if len(alpha) else 0.0,
        "scalar_edge_row_count": int(Aedge.shape[0]), "scalar_edge_nnz": int(Aedge.nnz),
        "edge_gradient": float(edge_gradient), "trust_mm": float(trust_mm),
        "source_cap_mm": float(source_cap_mm), "area_floor_for_actual_gate": float(area_floor),
        "linearized_cut_margin": float(cut_margin), "near_floor_band": float(near_band),
        "method": "maximize sum(alpha_i) under scalarized existing 408 edge rows, analytic per-node trust/source segment caps, and sequential local double-area tangent cuts",
        "optimality_claim": False, "native_admission": False, "accepted_physical_steps": 0,
    }
