"""Construction-only P1 exclusion constraints; exact final mesh gates remain mandatory."""
from __future__ import annotations
import numpy as np


def triangle_tet_barycentric_polygon(tetrahedron, triangle, *, condition_limit=1e12):
    """Clip a triangle against a tetrahedron, returning both parent barycentrics.

    Arrays use Float64 construction arithmetic. No physical source coordinates
    are changed. Empty intersection returns two (0, n) arrays. Boundary-only
    polygons are returned too, so touching obstacle cells are not silently missed.
    This is a construction helper, not an exact collision predicate.
    """
    X = np.asarray(tetrahedron, dtype=np.float64)
    P = np.asarray(triangle, dtype=np.float64)
    if X.shape != (4, 3) or P.shape != (3, 3):
        raise ValueError("expected tetrahedron (4,3) and triangle (3,3)")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(P)):
        raise ValueError("nonfinite clipping input")
    A = (X[1:] - X[0]).T
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("invalid condition limit")
    condition = float(np.linalg.cond(A))
    if not np.isfinite(condition) or condition > condition_limit:
        raise ValueError("degenerate or ill-conditioned parent tetrahedron")
    u = np.linalg.solve(A, (P - X[0]).T).T
    L = np.column_stack((1.0 - u.sum(axis=1), u))
    poly = [(L[i].copy(), np.eye(3)[i]) for i in range(3)]
    for k in range(4):
        out = []
        if not poly:
            break
        prev = poly[-1]
        a = float(prev[0][k])
        for cur in poly:
            b = float(cur[0][k])
            if (a < 0) != (b < 0):
                t = a / (a - b)
                lam = prev[0] + t * (cur[0] - prev[0])
                source = prev[1] + t * (cur[1] - prev[1])
                lam[k] = 0.0
                out.append((lam, source))
            if b >= 0:
                out.append(cur)
            prev, a = cur, b
        # Remove only repeated clipping vertices, not geometry or source faces.
        poly = []
        for item in out:
            if not poly or not (np.array_equal(item[0], poly[-1][0]) and
                                np.array_equal(item[1], poly[-1][1])):
                poly.append(item)
        if len(poly) > 1 and np.array_equal(poly[0][0], poly[-1][0]) and np.array_equal(poly[0][1], poly[-1][1]):
            poly.pop()
    if not poly:
        return np.empty((0, 4)), np.empty((0, 3))
    B = np.asarray([a for a, _ in poly])
    C = np.asarray([b for _, b in poly])
    if (np.any(B < -2e-12) or np.any(C < -2e-12) or
        np.max(np.abs(B.sum(axis=1) - 1)) > 2e-12 or
        np.max(np.abs(C.sum(axis=1) - 1)) > 2e-12):
        raise ValueError("clipping barycentric certificate failed")
    return B, C


def raise_exclusion_field(phi, tetrahedra, constraints, fixed_nodes, *, target=1.0,
                          maximum_value=1e6, tolerance=1e-12):
    """Monotonically raise free nodal field values to exclude convex polygons.

    constraints contains (tet_id, polygon_barycentrics) records. A P1 field
    attaining >=target on all polygon vertices attains it everywhere on that
    obstacle/tet intersection polygon. Fixed nodes are never altered. All
    constraints are rechecked after construction. Partial corrections with
    unresolvable fixed-boundary conflicts must not be admitted.
    """
    p = np.array(phi, dtype=np.float64, copy=True)
    T = np.asarray(tetrahedra)
    if p.ndim != 1 or not np.all(np.isfinite(p)):
        raise ValueError("invalid scalar field")
    if T.ndim != 2 or T.shape[1] != 4 or not np.issubdtype(T.dtype, np.integer):
        raise ValueError("invalid parent tetrahedra")
    if np.any(T < 0) or np.any(T >= len(p)):
        raise ValueError("parent node outside scalar field")
    fixed_ids = np.asarray(fixed_nodes)
    if fixed_ids.ndim != 1 or (len(fixed_ids) and not np.issubdtype(fixed_ids.dtype, np.integer)):
        raise ValueError("invalid fixed node ids")
    if np.any(fixed_ids < 0) or np.any(fixed_ids >= len(p)):
        raise ValueError("fixed node outside scalar field")
    if not np.isfinite(target) or not np.isfinite(maximum_value) or maximum_value <= target or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid construction limits")
    fixed = np.zeros(len(p), dtype=bool)
    fixed[fixed_ids.astype(np.int64)] = True
    rows = []
    for tet_id, B in constraints:
        ti = int(tet_id)
        B = np.asarray(B, dtype=np.float64)
        if ti != tet_id or ti < 0 or ti >= len(T) or B.ndim != 2 or B.shape[1] != 4:
            raise ValueError("invalid exclusion parent or barycentric shape")
        if (not np.all(np.isfinite(B)) or np.any(B < -tolerance) or
            (len(B) and np.max(np.abs(B.sum(axis=1) - 1)) > 2e-12)):
            raise ValueError("invalid exclusion barycentrics")
        # Roundoff-sized negative weights cannot provide a monotonic certificate.
        if np.any(B < 0):
            raise ValueError("negative exclusion barycentric weight")
        rows.extend((ti, b.copy()) for b in B)
    conflicts = []
    updated_rows = 0
    for row_id, (ti, b) in enumerate(rows):
        ids = T[ti]
        value = float(b @ p[ids])
        if value >= target - tolerance:
            continue
        free = ~fixed[ids]
        support = float(b[free].sum())
        if support <= tolerance:
            conflicts.append({"row": row_id, "tet_id": ti,
                              "reason": "fixed-boundary obstacle conflict", "value": value})
            continue
        increment = (target - value + tolerance) / support
        proposed = p[ids[free]] + increment
        if not np.all(np.isfinite(proposed)) or np.max(proposed) > maximum_value:
            conflicts.append({"row": row_id, "tet_id": ti,
                              "reason": "construction field cap", "required_increment": increment})
            continue
        p[ids[free]] = proposed
        updated_rows += 1
    values = np.asarray([b @ p[T[ti]] for ti, b in rows])
    failed = np.flatnonzero(values < target - tolerance)
    changed = np.flatnonzero(p != np.asarray(phi, dtype=np.float64))
    if np.any(p < phi) or not np.array_equal(p[fixed], np.asarray(phi)[fixed]):
        raise AssertionError("monotonic or fixed-boundary invariant failed")
    report = {"constraints": len(rows), "updated_rows": updated_rows,
              "changed_nodes": int(len(changed)), "failed_rows": failed.tolist(),
              "conflicts": conflicts, "minimum_obstacle_value": float(values.min()) if len(values) else None,
              "maximum_field_value": float(p.max()) if len(p) else None,
              "pass": len(failed) == 0}
    return p, report
