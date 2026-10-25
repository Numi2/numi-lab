"""Baseline SAT separators expressed as sparse mechanical-coordinate inequalities.

This module only builds linear rows for a joint mechanical fit. Passing these
rows does not certify triangle collision freedom or whole-surface clearance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy import sparse


class SurfaceSeparationError(ValueError):
    """Invalid geometry/transfer data or no valid baseline separator."""


@dataclass(frozen=True)
class PairSeparator:
    normal: np.ndarray
    axis_source: str
    baseline_min_gap_m: float
    margin_m: float
    candidate_min_gap_m: float | None
    candidate_violation_m: float
    projection_sensitivity: float
    shared_vertex_ids: tuple[int, ...]


def _finite_array(value: Any, shape: tuple[int, ...], name: str,
                  *, integer: bool = False) -> np.ndarray:
    a = np.asarray(value)
    if a.shape != shape:
        raise SurfaceSeparationError(f"{name} must have shape {shape}, got {a.shape}")
    if integer:
        if a.dtype.kind not in "iu":
            raise SurfaceSeparationError(f"{name} must contain integer IDs")
    elif a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise SurfaceSeparationError(f"{name} must be finite numeric data")
    return a


def _unit(v: np.ndarray, tol: float = 1e-14) -> np.ndarray | None:
    length = float(np.linalg.norm(v))
    if not np.isfinite(length) or length <= tol:
        return None
    return np.asarray(v, dtype=np.float64) / length


def sat_axes(triangle_a: np.ndarray, triangle_b: np.ndarray
             ) -> list[tuple[str, np.ndarray]]:
    """Return normalized triangle-normal, edge-cross, and in-plane SAT axes."""
    a = _finite_array(triangle_a, (3, 3), "triangle_a")
    b = _finite_array(triangle_b, (3, 3), "triangle_b")
    e_a = np.roll(a, -1, axis=0) - a
    e_b = np.roll(b, -1, axis=0) - b
    n_a = _unit(np.cross(e_a[0], e_a[1]))
    n_b = _unit(np.cross(e_b[0], e_b[1]))
    raw: list[tuple[str, np.ndarray]] = []
    if n_a is not None:
        raw.append(("triangle_normal_a", n_a))
    if n_b is not None:
        raw.append(("triangle_normal_b", n_b))
    for i, edge_a in enumerate(e_a):
        for j, edge_b in enumerate(e_b):
            raw.append((f"edge_cross_{i}_{j}", np.cross(edge_a, edge_b)))
    # These supply the in-plane axes for coplanar triangles. Extra axes are
    # safe too: each is retained only after direct interval-separation checks.
    for side, (edges, normal) in enumerate(((e_a, n_a), (e_b, n_b))):
        if normal is not None:
            for edge_i, edge in enumerate(edges):
                raw.append((f"in_plane_normal_cross_edge_{side}_{edge_i}",
                            np.cross(normal, edge)))
    axes: list[tuple[str, np.ndarray]] = []
    representatives: list[np.ndarray] = []
    for label, candidate in raw:
        axis = _unit(candidate)
        if axis is None:
            continue
        if any(abs(float(np.dot(axis, old))) >= 1.0 - 1e-12
               for old in representatives):
            continue
        representatives.append(axis)
        axes.append((label, axis))
    return axes


def _validate_transfer(weights: sparse.spmatrix, detail_count: int,
                       mechanical_count: int) -> sparse.csr_matrix:
    w = sparse.csr_matrix(weights, dtype=np.float64, copy=True)
    w.sum_duplicates()
    w.eliminate_zeros()
    w.sort_indices()
    if w.shape != (detail_count, mechanical_count):
        raise SurfaceSeparationError(
            f"weights shape must be {(detail_count, mechanical_count)}, got {w.shape}")
    if not np.isfinite(w.data).all() or np.any(w.data < 0.0):
        raise SurfaceSeparationError("weights must be finite and nonnegative")
    sums = np.asarray(w.sum(axis=1)).ravel()
    active = np.diff(w.indptr) > 0
    if np.any(np.abs(sums[active] - 1.0) > 1e-6):
        raise SurfaceSeparationError("nonempty transfer rows must sum to one")
    if np.any(sums[~active] != 0.0):
        raise SurfaceSeparationError("empty transfer rows must be exactly zero")
    return w


def select_baseline_separator(
    triangle_a: np.ndarray,
    triangle_b: np.ndarray,
    face_a_ids: np.ndarray,
    face_b_ids: np.ndarray,
    weights_a: sparse.spmatrix,
    weights_b: sparse.spmatrix,
    *,
    candidate_a: np.ndarray | None = None,
    candidate_b: np.ndarray | None = None,
    tolerance_m: float = 1e-12,
) -> PairSeparator:
    """Choose a baseline-valid weak separator, preferring margin/projection.

    If candidate triangles are supplied, a separator that exposes a candidate
    violation is preferred, then margin and transfer sensitivity. Shared
    indexed vertices force a zero pair margin.
    """
    a = _finite_array(triangle_a, (3, 3), "triangle_a").astype(np.float64)
    b = _finite_array(triangle_b, (3, 3), "triangle_b").astype(np.float64)
    ids_a = _finite_array(face_a_ids, (3,), "face_a_ids", integer=True).astype(np.int64)
    ids_b = _finite_array(face_b_ids, (3,), "face_b_ids", integer=True).astype(np.int64)
    if np.any(ids_a < 0) or np.any(ids_b < 0):
        raise SurfaceSeparationError("face vertex IDs must be nonnegative")
    if len(set(ids_a.tolist())) != 3 or len(set(ids_b.tolist())) != 3:
        raise SurfaceSeparationError("triangle face IDs must be distinct")
    wa = sparse.csr_matrix(weights_a, dtype=np.float64)
    wb = sparse.csr_matrix(weights_b, dtype=np.float64)
    if wa.shape[0] != 3 or wb.shape[0] != 3 or wa.shape[1] != wb.shape[1]:
        raise SurfaceSeparationError("face transfer rows must be 3×N with matching N")
    wa = _validate_transfer(wa, 3, wa.shape[1])
    wb = _validate_transfer(wb, 3, wb.shape[1])
    ca = None if candidate_a is None else _finite_array(candidate_a, (3, 3), "candidate_a").astype(np.float64)
    cb = None if candidate_b is None else _finite_array(candidate_b, (3, 3), "candidate_b").astype(np.float64)
    if (ca is None) != (cb is None):
        raise SurfaceSeparationError("candidate_a and candidate_b must be supplied together")
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise SurfaceSeparationError("tolerance_m must be finite and nonnegative")

    shared = tuple(sorted(set(ids_a.tolist()) & set(ids_b.tolist())))
    for i in range(3):
        for j in range(3):
            if ids_a[i] == ids_b[j]:
                if not np.array_equal(a[i], b[j]):
                    raise SurfaceSeparationError("shared vertex ID has different baseline coordinates")
                delta_w = (wa.getrow(i) - wb.getrow(j)).tocsr()
                delta_w.eliminate_zeros()
                if delta_w.nnz:
                    raise SurfaceSeparationError("shared vertex ID has different transfer rows")
                if ca is not None and not np.array_equal(ca[i], cb[j]):
                    raise SurfaceSeparationError("shared vertex ID has different candidate coordinates")
    cross_pairs = [(i, j) for i in range(3) for j in range(3)]
    nonshared_pairs = [(i, j) for i, j in cross_pairs if ids_a[i] != ids_b[j]]
    if not nonshared_pairs:
        raise SurfaceSeparationError("triangle pair has no nonshared vertex pair")
    choices = []
    for label, axis in sat_axes(a, b):
        for sign in (1.0, -1.0):
            normal = axis * sign
            all_gaps = [float(np.dot(normal, b[j] - a[i])) for i, j in cross_pairs]
            if min(all_gaps) < -tolerance_m:
                continue
            nonshared_gaps = [float(np.dot(normal, b[j] - a[i]))
                              for i, j in nonshared_pairs]
            available_gap = max(0.0, min(nonshared_gaps))
            margin = 0.0 if shared else (0.5 * available_gap if available_gap > tolerance_m else 0.0)
            sensitivity = max(
                (float(np.linalg.norm((wb.getrow(j) - wa.getrow(i)).data))
                 for i, j in nonshared_pairs), default=0.0)
            candidate_gap = None
            if ca is not None and cb is not None:
                candidate_gap = min(float(np.dot(normal, cb[j] - ca[i]))
                                    for i, j in nonshared_pairs)
            violation = 0.0 if candidate_gap is None else max(0.0, margin - candidate_gap)
            choices.append((violation > tolerance_m, margin, sensitivity,
                            violation, available_gap, label, normal,
                            candidate_gap))
    if not choices:
        raise SurfaceSeparationError("no valid baseline SAT separator")
    # Violation visibility is a tie-break gate; maximize margin/projection
    # among useful axes, then prefer the strongest witnessed violation.
    selected = max(choices, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
    return PairSeparator(
        normal=selected[6].copy(), axis_source=selected[5],
        baseline_min_gap_m=selected[4], margin_m=selected[1],
        candidate_min_gap_m=selected[7], candidate_violation_m=selected[3],
        projection_sensitivity=selected[2], shared_vertex_ids=shared)


def _vertex_constraint_row(
    vertex_a: int,
    vertex_b: int,
    normal: np.ndarray,
    margin_m: float,
    detail_origin: np.ndarray,
    transfer_origin: np.ndarray,
    mechanical_baseline: np.ndarray,
    weights_a: sparse.csr_matrix,
    weights_b: sparse.csr_matrix,
) -> tuple[sparse.csr_matrix | None, float | None, dict[str, Any]]:
    """Emit one absolute-coordinate inequality or validate an invariant row."""
    d = (weights_b - weights_a).tocsr()
    d.sum_duplicates()
    d.eliminate_zeros()
    source_gap = float(np.dot(normal, detail_origin[vertex_b] - detail_origin[vertex_a]))
    mechanical_gap = float(np.dot(normal, np.asarray(d @ (mechanical_baseline - transfer_origin)).reshape(3)))
    baseline_gap = source_gap + mechanical_gap
    if d.nnz == 0:
        if baseline_gap < margin_m - 1e-12:
            raise SurfaceSeparationError(
                f"fixed zero-transfer pair {vertex_a}/{vertex_b} fails margin: "
                f"gap={baseline_gap}, margin={margin_m}")
        return None, None, {"status": "omitted_fixed_transfer_zero_row",
                            "vertex_a": int(vertex_a), "vertex_b": int(vertex_b),
                            "baseline_gap_m": baseline_gap, "margin_m": float(margin_m),
                            "target": None}
    coo = d.tocoo()
    cols: list[int] = []
    vals: list[float] = []
    for col, value in zip(coo.col, coo.data):
        for component in range(3):
            coefficient = float(value * normal[component])
            if coefficient != 0.0:
                cols.append(3 * int(col) + component)
                vals.append(coefficient)
    if not cols:
        if baseline_gap < margin_m - 1e-12:
            raise SurfaceSeparationError("zero coefficient row violates requested margin")
        return None, None, {"status": "omitted_fixed_transfer_zero_row",
                            "vertex_a": int(vertex_a), "vertex_b": int(vertex_b),
                            "baseline_gap_m": baseline_gap, "margin_m": float(margin_m),
                            "target": None}
    n_mechanical = transfer_origin.shape[0]
    row = sparse.csr_matrix((vals, (np.zeros(len(cols), dtype=np.int64), cols)),
                             shape=(1, 3 * n_mechanical))
    target = float(margin_m - source_gap + (row @ transfer_origin.reshape(-1)).item())
    baseline_lhs = float((row @ mechanical_baseline.reshape(-1)).item())
    slack = baseline_lhs - target
    if slack < -1e-12:
        raise SurfaceSeparationError(
            f"baseline row {vertex_a}/{vertex_b} infeasible by {-slack} m")
    return row, target, {"status": "constraint", "vertex_a": int(vertex_a),
                         "vertex_b": int(vertex_b), "baseline_gap_m": baseline_gap,
                         "margin_m": float(margin_m), "target": target,
                         "baseline_slack_m": slack}


def build_pair_inequalities(
    face_a_ids: np.ndarray,
    face_b_ids: np.ndarray,
    detail_origin: np.ndarray,
    displacement_map: sparse.spmatrix,
    transfer_origin: np.ndarray,
    mechanical_baseline: np.ndarray,
    *,
    candidate_detail_positions: np.ndarray | None = None,
    pair_id: Any = None,
    tolerance_m: float = 1e-12,
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, Any]]:
    """Build the nine cross-triangle vertex inequalities via a 075-style W.

    Formula: C = n ⊗ (W_b - W_a), and
    C X >= margin - n·(detail_origin_b-detail_origin_a) + C transfer_origin.
    Shared indexed vertices are recorded and omitted as exact zero rows.
    """
    detail = np.asarray(detail_origin)
    if detail.ndim != 2 or detail.shape[1] != 3 or detail.dtype.kind not in "fiu" or not np.isfinite(detail).all():
        raise SurfaceSeparationError("detail_origin must be finite N×3 numeric data")
    n_detail = len(detail)
    xref = np.asarray(transfer_origin)
    xbase = np.asarray(mechanical_baseline)
    if xref.ndim != 2 or xref.shape[1] != 3 or xref.dtype.kind not in "fiu" or not np.isfinite(xref).all():
        raise SurfaceSeparationError("transfer_origin must be finite N×3 numeric data")
    if xbase.shape != xref.shape or xbase.dtype.kind not in "fiu" or not np.isfinite(xbase).all():
        raise SurfaceSeparationError("mechanical_baseline must match transfer_origin")
    w = _validate_transfer(displacement_map, n_detail, len(xref))
    ids_a = _finite_array(face_a_ids, (3,), "face_a_ids", integer=True).astype(np.int64)
    ids_b = _finite_array(face_b_ids, (3,), "face_b_ids", integer=True).astype(np.int64)
    if np.any(ids_a < 0) or np.any(ids_b < 0) or np.any(ids_a >= n_detail) or np.any(ids_b >= n_detail):
        raise SurfaceSeparationError("face vertex ID outside detailed vertex domain")
    if len(set(ids_a.tolist())) != 3 or len(set(ids_b.tolist())) != 3:
        raise SurfaceSeparationError("face IDs must be three distinct detailed vertices")
    candidate = None
    if candidate_detail_positions is not None:
        candidate = np.asarray(candidate_detail_positions)
        if candidate.shape != detail.shape or candidate.dtype.kind not in "fiu" or not np.isfinite(candidate).all():
            raise SurfaceSeparationError("candidate_detail_positions must be finite N×3")
    baseline = detail.astype(np.float64) + w @ (xbase.astype(np.float64) - xref.astype(np.float64))
    tri_a, tri_b = baseline[ids_a], baseline[ids_b]
    wa, wb = w[ids_a], w[ids_b]
    selected = select_baseline_separator(
        tri_a, tri_b, ids_a, ids_b, wa, wb,
        candidate_a=None if candidate is None else candidate[ids_a],
        candidate_b=None if candidate is None else candidate[ids_b],
        tolerance_m=tolerance_m)
    rows=[]; targets=[]; pair_provenance=[]
    for i in range(3):
        for j in range(3):
            va,vb=int(ids_a[i]),int(ids_b[j])
            if va==vb:
                if selected.margin_m != 0.0:
                    raise SurfaceSeparationError("shared indexed feature requires zero pair margin")
                if not np.array_equal(baseline[va], baseline[vb]):
                    raise SurfaceSeparationError("shared ID does not map to one baseline point")
                pair_provenance.append({"status":"omitted_shared_identity_zero_row",
                    "vertex_a":va,"vertex_b":vb,"baseline_gap_m":0.0,
                    "margin_m":0.0,"target":0.0})
                continue
            row,target,provenance=_vertex_constraint_row(
                va,vb,selected.normal,selected.margin_m,detail,xref,xbase,
                w.getrow(va),w.getrow(vb))
            if row is not None:
                rows.append(row);targets.append(float(target))
            pair_provenance.append(provenance)
    matrix=(sparse.vstack(rows,format="csr") if rows else
            sparse.csr_matrix((0,3*len(xref)),dtype=np.float64))
    matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
    targets_array=np.asarray(targets,dtype=np.float64)
    if matrix.shape[0] != len(targets_array):
        raise SurfaceSeparationError("row/target count mismatch")
    if len(targets_array):
        slack=np.asarray(matrix@xbase.reshape(-1)-targets_array)
        if not np.isfinite(slack).all() or float(slack.min()) < -1e-12:
            raise SurfaceSeparationError("baseline violates emitted pair inequalities")
    else:
        slack=np.empty(0,dtype=np.float64)
    record={
        "pair_id":pair_id,"axis_source":selected.axis_source,
        "normal":selected.normal.tolist(),
        "baseline_min_nonshared_gap_m":selected.baseline_min_gap_m,
        "margin_m":selected.margin_m,
        "candidate_min_nonshared_gap_m":selected.candidate_min_gap_m,
        "candidate_margin_violation_m":selected.candidate_violation_m,
        "projection_sensitivity":selected.projection_sensitivity,
        "shared_vertex_ids":list(selected.shared_vertex_ids),
        "constraint_row_count":int(matrix.shape[0]),
        "baseline_min_slack_m":None if not len(slack) else float(slack.min()),
        "vertex_pair_provenance":pair_provenance,
    }
    return matrix,targets_array,record

