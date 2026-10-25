"""Exact binary64 tetrahedron intersection classification for mass-transfer diagnostics.

This helper distinguishes full-dimensional overlap from lower-dimensional contact
in the exact rational geometry represented by input float values. It does not
replace tolerance-based clipping for volumes and is not a physical admission gate.
"""
from fractions import Fraction
import itertools
import numpy as np

def exact_binary64_intersection_rank(a_points, b_points):
    """Return exact affine rank of two binary64 tetrahedra's closed intersection.

    Input floats are interpreted exactly via Fraction.from_float. A full-dimensional
    convex intersection has rank 3; face/edge/point contacts have ranks 2/1/0;
    disjoint tetrahedra return rank -1. This is used only after the unchanged
    tolerance clip reports a positive opposite-material pair.
    """
    def to_fraction_tet(points):
        arr = np.asarray(points, dtype=np.float64)
        if arr.shape != (4, 3) or not np.all(np.isfinite(arr)):
            raise ValueError('exact intersection requires finite (4,3) tetrahedra')
        return [tuple(Fraction.from_float(float(x)) for x in row) for row in arr]

    def sub(a, b):
        return tuple(a[i] - b[i] for i in range(3))

    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1],
                a[2]*b[0]-a[0]*b[2],
                a[0]*b[1]-a[1]*b[0])

    def dot(a, b):
        return sum((a[i]*b[i] for i in range(3)), Fraction(0))

    def det3(m):
        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))

    def solve3(m, rhs):
        det = det3(m)
        if det == 0:
            return None
        out = []
        for col in range(3):
            replaced = [list(row) for row in m]
            for row in range(3):
                replaced[row][col] = rhs[row]
            out.append(det3(replaced) / det)
        return tuple(out)

    def tetra_planes(points):
        result = []
        for opposite in range(4):
            face_ids = [i for i in range(4) if i != opposite]
            p0, p1, p2 = (points[i] for i in face_ids)
            n = cross(sub(p1, p0), sub(p2, p0))
            h = dot(n, p0)
            side = dot(n, points[opposite]) - h
            if side == 0:
                raise ValueError('exact intersection input contains a degenerate tetrahedron')
            if side > 0:
                n = tuple(-x for x in n)
                h = -h
            result.append((n, h))  # closed interior halfspace n.x <= h
        return result

    a = to_fraction_tet(a_points)
    b = to_fraction_tet(b_points)
    planes = tetra_planes(a) + tetra_planes(b)
    vertices = set()
    for plane_ids in itertools.combinations(range(8), 3):
        matrix = [list(planes[i][0]) for i in plane_ids]
        rhs = [planes[i][1] for i in plane_ids]
        point = solve3(matrix, rhs)
        if point is None:
            continue
        if all(dot(n, point) <= h for n, h in planes):
            vertices.add(point)
    if not vertices:
        return {'affine_rank': -1, 'vertex_count': 0,
                'classification': 'exactly_disjoint'}

    points = list(vertices)
    origin = points[0]
    vectors = [sub(point, origin) for point in points[1:]]
    first = next((v for v in vectors if any(x != 0 for x in v)), None)
    if first is None:
        rank = 0
    else:
        normal = None
        for vector in vectors:
            candidate = cross(first, vector)
            if any(x != 0 for x in candidate):
                normal = candidate
                break
        if normal is None:
            rank = 1
        elif any(dot(normal, vector) != 0 for vector in vectors):
            rank = 3
        else:
            rank = 2
    classes = {
        0: 'exact_point_contact',
        1: 'exact_edge_contact',
        2: 'exact_face_contact',
        3: 'exact_positive_volume_intersection',
    }
    return {'affine_rank': rank, 'vertex_count': len(vertices),
            'classification': classes[rank]}

def classify_exact_cross_material_pair(a_points, b_points):
    result = exact_binary64_intersection_rank(a_points, b_points)
    rank = result['affine_rank']
    if rank == 3:
        action = 'retain_and_fail_closed_genuine_cross_material_overlap'
    elif rank == -1:
        action = 'omit_tolerance_only_positive_exactly_disjoint_pair'
    else:
        action = 'omit_zero_measure_pair_retain_boundary_contact_diagnostic'
    return {**result, 'transfer_action': action}

from math import fsum

def build_parent_local_mass_partition(
    child_indptr,
    parent_rows,
    overlap_volumes_m3,
    child_geometry_volumes_m3,
    parent_geometry_volumes_m3,
    child_material_ids,
    parent_material_ids,
    *,
    coverage_tolerance=1e-7,
):
    """Build a local parent-mass partition while preserving raw overlap geometry.

    Each accepted parent mass is apportioned among its same-material children by
    overlap / that parent's measured covered overlap. The overlap CSR is never
    changed. Before partitioning, child and parent coverage against their
    geometric volumes must satisfy the supplied unchanged coverage tolerance.
    This is a local roundoff partition, not a global mass rescale or geometry fix.
    """
    ptr_raw = np.asarray(child_indptr)
    rows_raw = np.asarray(parent_rows)
    if ptr_raw.ndim != 1 or not np.issubdtype(ptr_raw.dtype, np.integer):
        raise ValueError("child_indptr must be a one-dimensional integer array")
    if rows_raw.ndim != 1 or not np.issubdtype(rows_raw.dtype, np.integer):
        raise ValueError("parent_rows must be a one-dimensional integer array")
    ptr = ptr_raw.astype(np.int64, copy=False)
    rows = rows_raw.astype(np.int64, copy=False)
    overlaps = np.asarray(overlap_volumes_m3, dtype=np.float64)
    child_vol = np.asarray(child_geometry_volumes_m3, dtype=np.float64)
    parent_vol = np.asarray(parent_geometry_volumes_m3, dtype=np.float64)
    child_mat = np.asarray(child_material_ids)
    parent_mat = np.asarray(parent_material_ids)
    if not np.issubdtype(child_mat.dtype, np.integer) or not np.issubdtype(parent_mat.dtype, np.integer):
        raise ValueError("material IDs must be integer arrays")
    if np.any(child_mat <= 0) or np.any(parent_mat <= 0):
        raise ValueError("material IDs must be positive")
    if overlaps.ndim != 1 or overlaps.size != rows.size:
        raise ValueError("overlap volumes and parent rows must be one-dimensional and equal length")
    if child_vol.ndim != 1 or parent_vol.ndim != 1:
        raise ValueError("child and parent geometry volumes must be one-dimensional")
    if child_mat.shape != child_vol.shape or parent_mat.shape != parent_vol.shape:
        raise ValueError("material ID arrays must match their geometry-volume arrays")
    if ptr.shape != (len(child_vol) + 1,) or len(ptr) == 0:
        raise ValueError("child_indptr length must be child_count + 1")
    if ptr[0] != 0 or ptr[-1] != len(rows) or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError("child_indptr must be monotone from zero through the CSR entry count")
    if len(child_vol) == 0 or len(parent_vol) == 0:
        raise ValueError("empty child or parent domains are not valid mass partitions")
    if not np.all(np.isfinite(overlaps)) or np.any(overlaps <= 0):
        raise ValueError("every stored overlap volume must be finite and positive")
    if not np.all(np.isfinite(child_vol)) or np.any(child_vol <= 0):
        raise ValueError("child geometry volumes must be finite and positive")
    if not np.all(np.isfinite(parent_vol)) or np.any(parent_vol <= 0):
        raise ValueError("parent geometry volumes must be finite and positive")
    if not np.isfinite(coverage_tolerance) or coverage_tolerance < 0:
        raise ValueError("coverage_tolerance must be finite and nonnegative")
    if np.any(rows < 0) or np.any(rows >= len(parent_vol)):
        raise ValueError("parent row index is out of range")

    child_covered = np.empty(len(child_vol), dtype=np.float64)
    for ci in range(len(child_vol)):
        lo, hi = int(ptr[ci]), int(ptr[ci + 1])
        if len(set(int(v) for v in rows[lo:hi])) != hi - lo:
            raise ValueError(f"duplicate parent row in child CSR row {ci}")
        child_covered[ci] = fsum(float(v) for v in overlaps[lo:hi])
        if hi == lo:
            raise ValueError(f"child row {ci} has no positive overlap support")

    parent_entries = [[] for _ in range(len(parent_vol))]
    bad_material = []
    for ci in range(len(child_vol)):
        for k in range(int(ptr[ci]), int(ptr[ci + 1])):
            pi = int(rows[k])
            parent_entries[pi].append(float(overlaps[k]))
            if child_mat[ci] != parent_mat[pi]:
                cm = child_mat[ci].item() if hasattr(child_mat[ci], "item") else child_mat[ci]
                pm = parent_mat[pi].item() if hasattr(parent_mat[pi], "item") else parent_mat[pi]
                bad_material.append((ci, pi, cm, pm))
    if bad_material:
        raise ValueError(f"cross-material overlap is forbidden: {bad_material[:8]}")

    parent_covered = np.asarray([fsum(values) for values in parent_entries], dtype=np.float64)
    if np.any(parent_covered <= 0):
        raise ValueError(f"native parent rows have no positive overlap support: {np.flatnonzero(parent_covered <= 0)[:8].tolist()}")

    child_error = (child_covered - child_vol) / child_vol
    parent_error = (parent_covered - parent_vol) / parent_vol
    child_bad = np.flatnonzero(np.abs(child_error) > coverage_tolerance)
    parent_bad = np.flatnonzero(np.abs(parent_error) > coverage_tolerance)
    if child_bad.size or parent_bad.size:
        raise ValueError(
            "raw geometric coverage fails unchanged tolerance "
            f"{coverage_tolerance:g}: child rows {child_bad[:8].tolist()} "
            f"(max abs {float(np.max(np.abs(child_error))):.17g}), parent rows "
            f"{parent_bad[:8].tolist()} (max abs {float(np.max(np.abs(parent_error))):.17g})"
        )

    local_fraction = overlaps / parent_covered[rows]
    raw_geometric_fraction = overlaps / parent_vol[rows]
    correction = parent_vol / parent_covered
    return {
        "child_covered_volume_m3": child_covered,
        "parent_covered_volume_m3": parent_covered,
        "child_coverage_relative_error": child_error,
        "parent_coverage_relative_error": parent_error,
        "raw_parent_geometric_fraction_overlap_over_parent_volume": raw_geometric_fraction,
        "parent_local_mass_fraction_overlap_over_covered_volume": local_fraction,
        "parent_local_normalization_factor_geometry_over_covered": correction,
        "parent_local_correction_relative_to_raw": correction - 1.0,
        "maximum_absolute_parent_local_correction": float(np.max(np.abs(correction - 1.0))),
    }

