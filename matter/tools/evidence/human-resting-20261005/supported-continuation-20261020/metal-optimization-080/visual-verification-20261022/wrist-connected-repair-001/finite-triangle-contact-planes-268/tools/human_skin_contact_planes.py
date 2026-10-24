"""Canonical orientation and Jacobian helpers for barycentric contact rows."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def canonicalize_face_pair(face_a: int, face_b: int, bary_a, bary_b, normal):
    """Return one canonical face order and keep the oriented plane invariant.

    `normal` points from face B toward face A.  When the face order is
    canonicalized by swapping A and B, the barycentrics and normal must all be
    swapped/negated together.  The returned gap and derivative are then
    independent of the order in which an exact intersection predicate reports
    the pair.
    """
    fa, fb = int(face_a), int(face_b)
    ba = np.asarray(bary_a, dtype=np.float64).copy()
    bb = np.asarray(bary_b, dtype=np.float64).copy()
    n = np.asarray(normal, dtype=np.float64).copy()
    if ba.shape != (3,) or bb.shape != (3,) or n.shape != (3,):
        raise ValueError("contact barycentrics and normal must be length-three")
    if not np.all(np.isfinite(ba)) or not np.all(np.isfinite(bb)) or not np.all(np.isfinite(n)):
        raise ValueError("contact row contains a non-finite value")
    if fa == fb:
        raise ValueError("contact row requires two distinct face IDs")
    if fa > fb:
        fa, fb = fb, fa
        ba, bb = bb, ba
        n = -n
    return fa, fb, ba, bb, n


def assemble_barycentric_plane_row(
    face_a: int,
    face_b: int,
    vertices_a: Sequence[int],
    vertices_b: Sequence[int],
    bary_a,
    bary_b,
    normal,
    current_points_m,
    displacement_maps,
    free_lookup: Mapping[int, int],
):
    """Assemble the scalar plane gap and sparse derivative in mm variables.

    Each displacement variable is a registered-world millimetre vector.
    `displacement_maps[v]` maps that vector to the posed-world displacement.
    The returned coefficient dictionary is the derivative of the scalar gap in
    millimetres with respect to each scalar variable.
    """
    va = np.asarray(vertices_a, dtype=np.int64)
    vb = np.asarray(vertices_b, dtype=np.int64)
    if va.shape != (3,) or vb.shape != (3,):
        raise ValueError("contact triangles must have three vertex IDs")
    reverse = int(face_a) > int(face_b)
    fa, fb, ba, bb, n = canonicalize_face_pair(face_a, face_b, bary_a, bary_b, normal)
    if reverse:
        va, vb = vb, va
    points = np.asarray(current_points_m, dtype=np.float64)
    maps = np.asarray(displacement_maps, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or maps.shape != (len(points), 3, 3):
        raise ValueError("point/map shape mismatch")
    if np.any(va < 0) or np.any(vb < 0) or np.any(va >= len(points)) or np.any(vb >= len(points)):
        raise ValueError("contact triangle vertex ID out of range")
    if not np.isfinite(np.linalg.norm(n)) or np.linalg.norm(n) == 0:
        raise ValueError("contact plane normal must be finite and nonzero")

    pa = ba @ points[va]
    pb = bb @ points[vb]
    gap_mm = float(np.dot(pa - pb, n) * 1000.0)
    coeff = {}
    for tri, bary, sign in ((va, ba, 1.0), (vb, bb, -1.0)):
        for vertex, weight in zip(tri, bary):
            local = free_lookup.get(int(vertex))
            if local is None or abs(float(weight)) < 1e-14:
                continue
            grad = sign * float(weight) * (maps[int(vertex)].T @ n)
            for axis, value in enumerate(grad):
                key = 3 * int(local) + axis
                coeff[key] = coeff.get(key, 0.0) + float(value)
    return {
        "face_ids": (fa, fb),
        "bary_a": ba,
        "bary_b": bb,
        "normal": n,
        "gap_mm": gap_mm,
        "coeff": coeff,
        "point_a_m": pa,
        "point_b_m": pb,
    }


def finite_triangle_support_planes(triangle_a_m, triangle_b_m, preferred_normal=None):
    """Build finite-triangle separating-plane candidates for repair constraints.

    A returned normal points from B toward A. Its interval gap is
    ``min_i n.A_i - max_j n.B_j``. Enforcing every A/B corner inequality
    therefore separates the *entire* triangles, unlike moving their previous
    intersection points. For a fixed B, its maximum projection is a constant
    and only three A-corner rows are necessary. When both move, use all nine
    corner-pair rows with the normal frozen for that linearization.

    Candidates include triangle normals and every edge cross-product (the
    triangle SAT axes), plus in-plane edge normals for the coplanar case.
    Both orientations are considered. If supplied, preferred_normal restricts
    normals to its open hemisphere; callers can use a solid's outward normal
    to avoid resolving a crossing by pushing farther inside the solid.

    This floating-point construction is an optimization aid, never an exact
    collision or acceptance predicate. A new candidate still needs complete
    exact surface/closed-solid checks and nonlinear area/orientation checks.
    """
    a = np.asarray(triangle_a_m, dtype=np.float64)
    b = np.asarray(triangle_b_m, dtype=np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError("triangle coordinates must have shape (3,3)")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("triangle coordinates must be finite")

    def unit(vector):
        scale = float(np.max(np.abs(vector)))
        if scale == 0.0:
            return None
        if not np.isfinite(scale):
            raise ValueError("triangle edge/axis overflow")
        value = vector / scale
        return value / np.linalg.norm(value)

    ea = [unit(a[(k + 1) % 3] - a[k]) for k in range(3)]
    eb = [unit(b[(k + 1) % 3] - b[k]) for k in range(3)]
    if any(x is None for x in ea + eb):
        raise ValueError("triangle has a zero-length edge")
    na = unit(np.cross(ea[0], -ea[2]))
    nb = unit(np.cross(eb[0], -eb[2]))
    if na is None or nb is None:
        raise ValueError("triangle is degenerate")
    preferred = None
    if preferred_normal is not None:
        value = np.asarray(preferred_normal, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("preferred normal must be a finite three-vector")
        preferred = unit(value)
        if preferred is None:
            raise ValueError("preferred normal must be nonzero")
    axes = [("face_a", na), ("face_b", nb)]
    axes += [(f"edge_cross_{i}_{j}", unit(np.cross(e, f)))
             for i, e in enumerate(ea) for j, f in enumerate(eb)]
    axes += [(f"in_plane_a_{i}", unit(np.cross(na, e)))
             for i, e in enumerate(ea)]
    axes += [(f"in_plane_b_{i}", unit(np.cross(nb, e)))
             for i, e in enumerate(eb)]
    # Centering avoids cancellation from an arbitrary common world origin.
    origin = a[0] + (b[0] - a[0]) * 0.5
    ca, cb = a - origin, b - origin
    if not np.all(np.isfinite(ca)) or not np.all(np.isfinite(cb)):
        raise ValueError("triangle projection overflow")
    results = []
    for axis_index, (name, axis) in enumerate(axes):
        if axis is None:
            continue
        for sign in (1.0, -1.0):
            n = axis * sign
            if preferred is not None and float(n @ preferred) <= 0.0:
                continue
            pa, pb = ca @ n, cb @ n
            ia, ib = int(np.argmin(pa)), int(np.argmax(pb))
            gap = float((pa[ia] - pb[ib]) * 1000.0)
            if not np.isfinite(gap):
                raise ValueError("triangle interval gap overflow")
            results.append({
                "normal": n, "gap_mm": gap,
                "a_support": ia, "b_support": ib,
                "a_projection_mm": pa * 1000.0,
                "b_projection_mm": pb * 1000.0,
                "origin_m": origin.copy(), "axis": name,
                "axis_index": axis_index, "orientation": int(sign),
            })
    # Stable ties preserve the axis enumeration, for deterministic replay.
    return sorted(results, key=lambda row: -row["gap_mm"])
