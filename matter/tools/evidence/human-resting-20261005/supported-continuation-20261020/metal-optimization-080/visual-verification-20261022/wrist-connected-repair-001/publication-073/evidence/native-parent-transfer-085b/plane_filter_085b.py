"""Batched 073 exact clipping for surviving tetra pairs; 085 remains frozen reference."""
import itertools
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from plane_filter_085 import (
    INSIDE_TOL, UNIQUE_TOL, halfspaces, prepare_tetra, separated_prepared,
)

_COMBINATIONS = np.asarray(list(itertools.combinations(range(8), 3)), dtype=np.int8)
_DET_CUTOFF = 1e-10


def tetra_intersection_volume_stacked(a, b, inside_tol=INSIDE_TOL, unique_tol=UNIQUE_TOL):
    """073 clip with its 56 3x3 systems submitted as one stacked solve."""
    na, ba = halfspaces(a); nb, bb = halfspaces(b)
    if na is None or nb is None:
        return 0.0
    n = np.vstack((na, nb))
    h = np.r_[ba, bb]
    matrices = n[_COMBINATIONS]
    rhs = h[_COMBINATIONS]
    dets = np.linalg.det(matrices)

    # Preserve the exact scalar determinant decision near the owner cutoff.
    # Stacked determinant remains the fast path for all other triples.
    near = np.abs(np.abs(dets) - _DET_CUTOFF) <= 1e-14
    for i in np.flatnonzero(near):
        dets[i] = np.linalg.det(matrices[i])
    keep = np.abs(dets) >= _DET_CUTOFF
    if not np.any(keep):
        return 0.0
    try:
        solved = np.linalg.solve(matrices[keep], rhs[keep, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        # Match the scalar owner's failure behavior per triple.
        rows = []
        keep_ids = np.flatnonzero(keep)
        for i in keep_ids:
            try:
                rows.append(np.linalg.solve(matrices[i], rhs[i]))
            except np.linalg.LinAlgError:
                rows.append(None)
        valid_ids = [i for i, x in enumerate(rows) if x is not None]
        if not valid_ids:
            return 0.0
        keep_ids = keep_ids[valid_ids]
        solved = np.asarray([rows[i] for i in valid_ids])

    pts = []
    for x in solved:
        if np.max(n @ x - h) <= inside_tol:
            if not any(np.max(np.abs(x-y)) < unique_tol for y in pts):
                pts.append(x)
    if len(pts) < 4:
        return 0.0
    try:
        return float(ConvexHull(np.asarray(pts)).volume)
    except QhullError:
        return 0.0


def filtered_tetra_intersection_volume_stacked(a, b, inside_tol=INSIDE_TOL,
                                               unique_tol=UNIQUE_TOL,
                                               prepared_a=None, prepared_b=None):
    """085 conservative separator followed by the stacked exact 073 clip."""
    pa = prepare_tetra(a, inside_tol) if prepared_a is None else prepared_a
    pb = prepare_tetra(b, inside_tol) if prepared_b is None else prepared_b
    if separated_prepared(pa, pb, inside_tol):
        return 0.0, True
    return tetra_intersection_volume_stacked(a, b, inside_tol, unique_tol), False


def filtered_pairs_stacked(a_vertices, b_vertices, a_rows, b_rows,
                           prepared_a, prepared_b, inside_tol=INSIDE_TOL,
                           unique_tol=UNIQUE_TOL):
    """Evaluate paired rows with precomputed planes; no shared state is mutated."""
    aa = np.asarray(a_rows, dtype=np.int64)
    bb = np.asarray(b_rows, dtype=np.int64)
    if aa.shape != bb.shape or aa.ndim != 1:
        raise ValueError("paired row arrays must be same-length vectors")
    result = np.zeros(len(aa), dtype=np.float64)
    rejected = np.zeros(len(aa), dtype=bool)
    for k, (i, j) in enumerate(zip(aa.tolist(), bb.tolist())):
        result[k], rejected[k] = filtered_tetra_intersection_volume_stacked(
            a_vertices[i], b_vertices[j], inside_tol, unique_tol,
            prepared_a[i], prepared_b[j])
    return result, rejected

