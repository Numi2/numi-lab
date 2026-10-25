"""Exact, tolerance-preserving early rejection for the frozen 073 tetra overlap clip.

The filter is conservative: it rejects only when the two convex halfspace sets,
each expanded by the exact inside tolerance used by the owner, are provably
separated by one of the original face planes. Survivors use the unchanged clip.
"""
import itertools
import numpy as np
from scipy.spatial import ConvexHull, QhullError

INSIDE_TOL = 2e-10
UNIQUE_TOL = 2e-10


def tet_volume(v):
    return abs(float(np.linalg.det(v[1:] - v[0]))) / 6.0


def halfspaces(v):
    ns, bs = [], []
    for k in range(4):
        ids = [j for j in range(4) if j != k]
        p0, p1, p2 = v[ids]
        n = np.cross(p1-p0, p2-p0)
        ln = np.linalg.norm(n)
        if not np.isfinite(ln) or ln == 0:
            return None, None
        n /= ln
        if np.dot(n, v[k]-p0) > 0:
            n = -n
        ns.append(n)
        bs.append(np.dot(n, p0))
    return np.asarray(ns), np.asarray(bs)


def tetra_intersection_volume(a, b, inside_tol=INSIDE_TOL, unique_tol=UNIQUE_TOL):
    """Frozen 073 halfspace-triple intersection and ConvexHull volume calculation."""
    na, ba = halfspaces(a); nb, bb = halfspaces(b)
    if na is None or nb is None:
        return 0.0
    N = np.vstack((na, nb)); h = np.r_[ba, bb]
    pts = []
    for ix in itertools.combinations(range(8), 3):
        M = N[list(ix)]
        if abs(float(np.linalg.det(M))) < 1e-10:
            continue
        x = np.linalg.solve(M, h[list(ix)])
        if np.max(N @ x - h) <= inside_tol:
            if not any(np.max(np.abs(x-y)) < unique_tol for y in pts):
                pts.append(x)
    if len(pts) < 4:
        return 0.0
    try:
        return float(ConvexHull(np.asarray(pts)).volume)
    except QhullError:
        return 0.0


def _expanded_tetra_vertices(v, tol):
    """Vertices of the tetra's four unit-plane inequalities shifted outward by tol.

    Returns None if the shifted halfspace intersection is numerically ill-conditioned;
    callers must then fall back to the exact clip.
    """
    ns, bs = halfspaces(v)
    if ns is None:
        return None
    rhs = bs + tol
    verts = []
    scale = max(1.0, float(np.max(np.abs(v))), float(np.max(np.abs(rhs))))
    feasibility_guard = 128.0 * np.finfo(np.float64).eps * scale
    for ix in itertools.combinations(range(4), 3):
        M = ns[list(ix)]
        det = float(np.linalg.det(M))
        if not np.isfinite(det) or abs(det) < 1e-10:
            return None
        try:
            x = np.linalg.solve(M, rhs[list(ix)])
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(x)) or float(np.max(ns @ x - rhs)) > feasibility_guard:
            return None
        verts.append(x)
    if len(verts) != 4:
        return None
    return np.asarray(verts)


def separated_by_owner_planes(a, b, inside_tol=INSIDE_TOL):
    """Return True only if the 073 clip's tolerance-expanded convex sets are disjoint."""
    if not np.isfinite(inside_tol) or inside_tol < 0:
        raise ValueError("inside_tol must be finite and nonnegative")
    na, ba = halfspaces(a); nb, bb = halfspaces(b)
    if na is None or nb is None:
        return False
    va = _expanded_tetra_vertices(a, inside_tol)
    vb = _expanded_tetra_vertices(b, inside_tol)
    if va is None or vb is None:
        return False

    coordinate_scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))),
                          float(np.max(np.abs(ba))), float(np.max(np.abs(bb))))
    roundoff_guard = 256.0 * np.finfo(np.float64).eps * coordinate_scale

    # For each B plane, the minimum of its signed violation over expanded A
    # is attained at an expanded-A vertex. A strict positive minimum beyond
    # B's same tolerance proves the two expanded sets disjoint.
    min_a_against_b = np.min(nb @ va.T, axis=1) - bb
    if np.any(min_a_against_b > inside_tol + roundoff_guard):
        return True
    min_b_against_a = np.min(na @ vb.T, axis=1) - ba
    return bool(np.any(min_b_against_a > inside_tol + roundoff_guard))


def filtered_tetra_intersection_volume(a, b, inside_tol=INSIDE_TOL, unique_tol=UNIQUE_TOL):
    if separated_by_owner_planes(a, b, inside_tol):
        return 0.0, True
    return tetra_intersection_volume(a, b, inside_tol, unique_tol), False


def prepare_tetra(v, inside_tol=INSIDE_TOL):
    """Precompute owner planes and their tolerance-expanded vertices for batch tests."""
    if not np.isfinite(inside_tol) or inside_tol < 0:
        raise ValueError("inside_tol must be finite and nonnegative")
    ns, bs = halfspaces(np.asarray(v, dtype=np.float64))
    if ns is None:
        return None
    expanded = _expanded_tetra_vertices(np.asarray(v, dtype=np.float64), inside_tol)
    if expanded is None:
        return None
    return ns, bs, expanded


def separated_prepared(pa, pb, inside_tol=INSIDE_TOL):
    """Scalar predicate using prepared tetrahedra; False means exact clip required."""
    if pa is None or pb is None:
        return False
    na, ba, va = pa
    nb, bb, vb = pb
    coordinate_scale = max(1.0, float(np.max(np.abs(va))), float(np.max(np.abs(vb))),
                          float(np.max(np.abs(ba))), float(np.max(np.abs(bb))))
    guard = 256.0 * np.finfo(np.float64).eps * coordinate_scale
    if np.any(np.min(nb @ va.T, axis=1) - bb > inside_tol + guard):
        return True
    return bool(np.any(np.min(na @ vb.T, axis=1) - ba > inside_tol + guard))


def separated_pairs_prepared(normals_a, offsets_a, expanded_a, valid_a,
                             normals_b, offsets_b, expanded_b, valid_b,
                             rows_a, rows_b, inside_tol=INSIDE_TOL):
    """Vectorized conservative rejection for paired rows, suitable for bounded batches."""
    ai = np.asarray(rows_a, dtype=np.int64)
    bi = np.asarray(rows_b, dtype=np.int64)
    if ai.shape != bi.shape or ai.ndim != 1:
        raise ValueError("paired row arrays must be same-length vectors")
    n = len(ai)
    rejected = np.zeros(n, dtype=bool)
    if n == 0:
        return rejected
    usable = np.asarray(valid_a, dtype=bool)[ai] & np.asarray(valid_b, dtype=bool)[bi]
    pos = np.flatnonzero(usable)
    if len(pos) == 0:
        return rejected
    a = ai[pos]; b = bi[pos]
    na = normals_a[a]; ba = offsets_a[a]; va = expanded_a[a]
    nb = normals_b[b]; bb = offsets_b[b]; vb = expanded_b[b]
    min_a_against_b = np.min(np.einsum("pfc,pvc->pfv", nb, va, optimize=True), axis=2) - bb
    min_b_against_a = np.min(np.einsum("pfc,pvc->pfv", na, vb, optimize=True), axis=2) - ba
    scale = np.maximum(1.0, np.maximum(np.max(np.abs(va), axis=(1, 2)),
                                       np.max(np.abs(vb), axis=(1, 2))))
    guard = 256.0 * np.finfo(np.float64).eps * scale[:, None]
    local = (np.any(min_a_against_b > inside_tol + guard, axis=1) |
             np.any(min_b_against_a > inside_tol + guard, axis=1))
    rejected[pos] = local
    return rejected
