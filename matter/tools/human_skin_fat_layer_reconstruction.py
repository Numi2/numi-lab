"""Deterministic O/I/D layer reconstruction for connected surface authoring.

This static surface construction helper does not create or admit volume
cells, material maps, native runtime state, or physical steps.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve

# Informational warning only; this threshold never rejects a positive cell.
SPLIT_QUALITY_WARNING_THRESHOLD = 1.0e-6


@dataclass(frozen=True)
class LayerReconstruction:
    outer_points: np.ndarray
    inner_points: np.ndarray
    deep_points: np.ndarray
    outward_normals: np.ndarray
    inner_displacement: np.ndarray
    deep_displacement: np.ndarray
    active_vertex_mask: np.ndarray
    diagnostics: dict


def _finite_points(value, name, *, rows=None):
    x = np.asarray(value)
    if x.ndim != 2 or x.shape[1] != 3 or (rows is not None and x.shape[0] != rows):
        raise ValueError(f"{name} must have shape ({rows if rows is not None else 'N'}, 3)")
    x = x.astype(np.float64, copy=False)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite")
    return x


def _positive_depth_field(value, name, rows):
    """Return a finite positive per-vertex depth field from scalar or (N,) input."""
    depth = np.asarray(value, dtype=np.float64)
    if depth.ndim == 0:
        depth = np.full(rows, float(depth), dtype=np.float64)
    elif depth.ndim != 1 or depth.shape != (rows,):
        raise ValueError(f"{name} must be a finite positive scalar or have shape ({rows},)")
    if not np.isfinite(depth).all() or np.any(depth <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return depth


def _vertex_normals(points, faces, orientation_sign, active):
    tri = points[faces]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas2 = np.linalg.norm(raw, axis=1)
    if not np.isfinite(areas2).all() or np.any(areas2 <= 0.0):
        rows = np.flatnonzero(~np.isfinite(areas2) | (areas2 <= 0.0))[:16].tolist()
        raise ValueError(f"degenerate or nonfinite outer faces at rows {rows}")
    raw *= float(orientation_sign)
    normals = np.zeros_like(points, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], raw)
    norm = np.linalg.norm(normals, axis=1)
    bad = np.flatnonzero(active & (~np.isfinite(norm) | (norm <= np.finfo(float).tiny)))
    if bad.size:
        raise ValueError(f"active vertices have no oriented normal: {bad[:16].tolist()}")
    valid = np.isfinite(norm) & (norm > np.finfo(float).tiny)
    normals[valid] /= norm[valid, None]
    return normals


def _active_graph(n, faces, active_indices):
    active_mask = np.zeros(n, dtype=bool)
    active_mask[active_indices] = True
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    edges = edges[active_mask[edges[:, 0]] & active_mask[edges[:, 1]]]
    loc = np.full(n, -1, dtype=np.int64)
    loc[active_indices] = np.arange(len(active_indices), dtype=np.int64)
    if len(edges):
        i = loc[edges[:, 0]]
        j = loc[edges[:, 1]]
        row = np.concatenate((i, j))
        col = np.concatenate((j, i))
        adj = sparse.coo_matrix((np.ones(len(row), dtype=np.float64), (row, col)),
                                shape=(len(active_indices), len(active_indices))).tocsr()
        adj.data[:] = 1.0
    else:
        adj = sparse.csr_matrix((len(active_indices), len(active_indices)), dtype=np.float64)
    return active_mask, loc, adj


def _harmonic_extension(active_indices, boundary_indices, boundary_values, adjacency, global_to_local):
    if len(active_indices) == 0:
        return np.zeros((0, 3), dtype=np.float64), 0
    boundary_local = global_to_local[boundary_indices]
    if np.any(boundary_local < 0):
        raise ValueError("fixed boundary vertices must be inside the active patch")
    if len(boundary_local) == 0:
        raise ValueError("active patch has no fixed boundary; refusing unconstrained harmonic solve")
    component_count, labels = connected_components(adjacency, directed=False, return_labels=True)
    anchored = np.zeros(component_count, dtype=bool)
    anchored[np.unique(labels[boundary_local])] = True
    if not anchored.all():
        missing = np.flatnonzero(~anchored).tolist()
        raise ValueError(f"active patch has components without fixed boundary: {missing}")
    lap = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    result = np.zeros((len(active_indices), 3), dtype=np.float64)
    result[boundary_local] = boundary_values
    is_boundary = np.zeros(len(active_indices), dtype=bool)
    is_boundary[boundary_local] = True
    free_local = np.flatnonzero(~is_boundary)
    if len(free_local):
        Lff = lap[free_local][:, free_local].tocsc()
        Lfb = lap[free_local][:, boundary_local].tocsr()
        rhs = -(Lfb @ boundary_values)
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            for axis in range(3):
                result[free_local, axis] = spsolve(Lff, rhs[:, axis])
        if not np.isfinite(result[free_local]).all():
            raise ValueError("harmonic solve returned nonfinite displacement")
    return result, int(component_count)


def reconstruct_connected_layers(
    outer_points,
    outer_faces,
    baseline_inner_offsets,
    baseline_deep_offsets,
    fixed_boundary_indices,
    boundary_outer_points,
    boundary_inner_points,
    boundary_deep_points,
    *,
    skin_depth_m: float,
    fat_depth_m: float,
    orientation_sign: int,
    active_vertex_indices=None,
):
    """Rebuild I/D shells from oriented O normals and anchored graph harmonics.

    Offsets define the unchanged baseline as I_base=O+di and D_base=I_base+dd.
    Vertices outside active_vertex_indices are copied from that baseline. Active
    vertices start at requested positive scalar or per-vertex normal depths, then receive
    a uniform-edge graph-harmonic displacement matching exact I/D boundary
    coordinates. Every active graph component must contain a fixed boundary
    vertex; no unconstrained component is silently solved.
    """
    P = _finite_points(outer_points, "outer_points")
    n = len(P)
    if n == 0:
        raise ValueError("outer_points must be nonempty")
    F = np.asarray(outer_faces)
    if F.ndim != 2 or F.shape[1] != 3 or F.dtype.kind not in "iu" or len(F) == 0:
        raise ValueError("outer_faces must be a nonempty integer Mx3 array")
    F = F.astype(np.int64, copy=False)
    if F.min(initial=0) < 0 or F.max(initial=0) >= n:
        raise ValueError("outer_faces contain out-of-range vertex rows")
    if orientation_sign not in (-1, 1):
        raise ValueError("orientation_sign must be +1 or -1")
    skin_depth = _positive_depth_field(skin_depth_m, "skin_depth_m", n)
    fat_depth = _positive_depth_field(fat_depth_m, "fat_depth_m", n)
    di = _finite_points(baseline_inner_offsets, "baseline_inner_offsets", rows=n)
    dd = _finite_points(baseline_deep_offsets, "baseline_deep_offsets", rows=n)
    baseline_I = P + di
    baseline_D = baseline_I + dd

    boundary = np.asarray(fixed_boundary_indices)
    if boundary.ndim != 1 or boundary.dtype.kind not in "iu":
        raise ValueError("fixed_boundary_indices must be a one-dimensional integer array")
    boundary = boundary.astype(np.int64, copy=False)
    if len(np.unique(boundary)) != len(boundary):
        raise ValueError("fixed_boundary_indices contain duplicates")
    if boundary.size and (boundary.min() < 0 or boundary.max() >= n):
        raise ValueError("fixed_boundary_indices contain out-of-range rows")
    bO = _finite_points(boundary_outer_points, "boundary_outer_points", rows=len(boundary))
    bI = _finite_points(boundary_inner_points, "boundary_inner_points", rows=len(boundary))
    bD = _finite_points(boundary_deep_points, "boundary_deep_points", rows=len(boundary))
    if not np.array_equal(bO, P[boundary]):
        raise ValueError("outer boundary coordinates must exactly match the supplied outer mesh")

    active = (np.arange(n, dtype=np.int64) if active_vertex_indices is None
              else np.asarray(active_vertex_indices))
    if active.ndim != 1 or active.dtype.kind not in "iu":
        raise ValueError("active_vertex_indices must be a one-dimensional integer array")
    active = active.astype(np.int64, copy=False)
    if len(np.unique(active)) != len(active):
        raise ValueError("active_vertex_indices contain duplicates")
    if active.size and (active.min() < 0 or active.max() >= n):
        raise ValueError("active_vertex_indices contain out-of-range rows")
    if not np.isin(boundary, active).all():
        raise ValueError("fixed boundary vertices must be part of active_vertex_indices")

    active_mask, global_to_local, adjacency = _active_graph(n, F, active)
    normals = _vertex_normals(P, F, orientation_sign, active_mask)
    O = P.copy()
    O[boundary] = bO
    I = baseline_I.copy()
    D = baseline_D.copy()
    I0 = O - skin_depth[:, None] * normals
    D0 = I0 - fat_depth[:, None] * normals
    if len(active):
        harm_I, components = _harmonic_extension(
            active, boundary, bI - I0[boundary], adjacency, global_to_local)
        harm_D, components_D = _harmonic_extension(
            active, boundary, bD - D0[boundary], adjacency, global_to_local)
        if components != components_D:
            raise RuntimeError("I/D harmonic component count mismatch")
        I[active] = I0[active] + harm_I
        D[active] = D0[active] + harm_D
    else:
        components = 0
    # Exact Dirichlet assignment follows solving so fixed rows never inherit solve error.
    O[boundary] = bO
    I[boundary] = bI
    D[boundary] = bD

    skin_proj = np.einsum("ij,ij->i", O - I, normals)
    fat_proj = np.einsum("ij,ij->i", I - D, normals)
    outside = ~active_mask
    diagnostics = {
        "status": "CONSTRUCTION_ONLY",
        "physical_or_native_admission": False,
        "vertex_count": int(n),
        "outer_face_count": int(len(F)),
        "active_vertex_count": int(len(active)),
        "fixed_boundary_count": int(len(boundary)),
        "active_graph_component_count": int(components),
        "outside_patch_preserved_from_baseline_offsets": bool(
            np.array_equal(I[outside], baseline_I[outside]) and
            np.array_equal(D[outside], baseline_D[outside]) and
            np.array_equal(O[outside], P[outside])),
        "outer_boundary_exact": bool(np.array_equal(O[boundary], bO)),
        "inner_boundary_exact": bool(np.array_equal(I[boundary], bI)),
        "deep_boundary_exact": bool(np.array_equal(D[boundary], bD)),
        "max_inner_displacement_from_baseline_m": float(np.max(np.linalg.norm(I - baseline_I, axis=1), initial=0.0)),
        "max_deep_displacement_from_baseline_m": float(np.max(np.linalg.norm(D - baseline_D, axis=1), initial=0.0)),
        "active_skin_normal_depth_min_m": float(np.min(skin_proj[active], initial=np.inf)),
        "active_skin_normal_depth_max_m": float(np.max(skin_proj[active], initial=-np.inf)),
        "active_fat_normal_depth_min_m": float(np.min(fat_proj[active], initial=np.inf)),
        "active_fat_normal_depth_max_m": float(np.max(fat_proj[active], initial=-np.inf)),
        "requested_skin_depth_min_m": float(np.min(skin_depth, initial=np.inf)),
        "requested_skin_depth_max_m": float(np.max(skin_depth, initial=-np.inf)),
        "requested_fat_depth_min_m": float(np.min(fat_depth, initial=np.inf)),
        "requested_fat_depth_max_m": float(np.max(fat_depth, initial=-np.inf)),
        "source_ancestry_modified": False,
        "volume_mesh_or_material_map_created": False,
    }
    return LayerReconstruction(
        outer_points=O, inner_points=I, deep_points=D,
        outward_normals=normals,
        inner_displacement=I - baseline_I,
        deep_displacement=D - baseline_D,
        active_vertex_mask=active_mask,
        diagnostics=diagnostics,
    )



@dataclass(frozen=True)
class ScalarFieldVolumeSplit:
    """Static conforming tetrahedral split induced by a nodal linear field.

    ``tetrahedra`` is a single index buffer with ``side_signs`` (-1 or +1),
    ``parent_cell_rows`` and optional inherited ``material_ids`` alongside it.
    ``interpolation_weights`` maps every output vertex to original source rows.
    This is offline construction data; it does not alter a runtime or admit a
    physical state.
    """
    points: np.ndarray
    points_float32: np.ndarray
    tetrahedra: np.ndarray
    parent_cell_rows: np.ndarray
    side_signs: np.ndarray
    material_ids: np.ndarray | None
    cell_masses: np.ndarray | None
    interpolation_weights: sparse.csr_matrix
    scalar_values: np.ndarray
    global_vertex_order_keys: tuple
    intersection_edge_global_ids: np.ndarray
    retained_parent_volumes: np.ndarray
    excluded_parent_volumes: np.ndarray
    retained_parent_masses: np.ndarray | None
    excluded_parent_masses: np.ndarray | None
    diagnostics: dict


def _tet_det(points, tet):
    p = points[np.asarray(tet, dtype=np.int64)]
    return float(np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0])))


def _strictly_opposite(a, b):
    # Avoid multiplication: tiny values can underflow and large values can
    # overflow even though only the signs are needed.
    return (a < 0.0 < b) or (b < 0.0 < a)


def _polygon_area_vector(points, polygon):
    q = np.asarray(points, dtype=np.float64)[np.asarray(polygon, dtype=np.int64)]
    if len(q) < 3:
        return np.zeros(3, dtype=np.float64)
    # Translate before the cross-product sum. With tiny clipped facets at
    # world-scale coordinates, summing cross products of absolute positions
    # can cancel to zero even when the local polygon area is positive.
    q = q - q[0]
    return np.sum(np.cross(q, np.roll(q, -1, axis=0)), axis=0)


def _clean_cycle(rows):
    out = []
    for r in rows:
        r = int(r)
        if not out or out[-1] != r:
            out.append(r)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    # A repeated nonconsecutive vertex makes a clipping cycle ambiguous.
    if len(set(out)) != len(out):
        raise ValueError("clipped facet contains a repeated nonconsecutive vertex")
    return out


def _pulling_polygon_triangles(polygon, order_key):
    """Fan a convex cyclic polygon from its globally least vertex."""
    p = _clean_cycle(polygon)
    if len(p) < 3:
        return []
    k = min(range(len(p)), key=lambda i: order_key[p[i]])
    p = p[k:] + p[:k]
    return [(p[0], p[i], p[i + 1]) for i in range(1, len(p) - 1)]


def _clip_oriented_triangle(face, scalar, side, intersection_row):
    """Sutherland-Hodgman clip of an oriented triangle by side*phi >= 0."""
    f = [int(x) for x in face]
    out = []
    for i, a in enumerate(f):
        b = f[(i + 1) % len(f)]
        va = float(scalar[a])
        vb = float(scalar[b])
        ina = side * va >= 0.0
        inb = side * vb >= 0.0
        if ina and inb:
            out.append(b)
        elif ina and not inb:
            out.append(intersection_row(a, b))
        elif not ina and inb:
            out.append(intersection_row(a, b))
            out.append(b)
    return _clean_cycle(out)


def _ordered_cut_polygon(rows, points, scalar, tet, side, order_key):
    """Return cut vertices in cyclic order with the clipped cell's outward normal."""
    rows = list(dict.fromkeys(int(r) for r in rows))
    if len(rows) < 3:
        raise ValueError("a mixed-sign tetrahedron produced fewer than three cut vertices")
    x = points[np.asarray(tet, dtype=np.int64)]
    dv = np.asarray(scalar, dtype=np.float64)[np.asarray(tet, dtype=np.int64)]
    # The cut geometry is invariant to scalar rescaling. Normalize locally so
    # a valid tiny-magnitude field does not underflow in the gradient norm.
    dv_scale = float(np.max(np.abs(dv), initial=0.0))
    if dv_scale == 0.0 or not np.isfinite(dv_scale):
        raise ValueError("mixed-sign tetrahedron has an invalid scalar gradient")
    dv = dv / dv_scale
    A = x[1:] - x[0]
    grad = np.linalg.solve(A, dv[1:] - dv[0])
    n = -float(side) * grad  # outward from the kept half-space
    nn = float(np.linalg.norm(n))
    if not np.isfinite(nn) or nn <= np.finfo(float).tiny:
        raise ValueError("mixed-sign tetrahedron has an invalid scalar gradient")
    n /= nn
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmin(np.abs(n)))] = 1.0
    u = np.cross(n, axis)
    un = float(np.linalg.norm(u))
    if not np.isfinite(un) or un <= np.finfo(float).tiny:
        raise ValueError("could not construct a stable cut-plane basis")
    u /= un
    v = np.cross(n, u)
    q = points[np.asarray(rows, dtype=np.int64)]
    center = np.mean(q, axis=0)
    rel = q - center
    angle = np.arctan2(rel @ v, rel @ u)
    ordered = sorted(range(len(rows)), key=lambda i: (float(angle[i]), order_key[rows[i]]))
    result = [rows[i] for i in ordered]
    if float(np.dot(_polygon_area_vector(points, result), n)) < 0.0:
        result.reverse()
    if float(np.dot(_polygon_area_vector(points, result), n)) <= 0.0:
        raise ValueError("cut polygon is degenerate or has inconsistent orientation")
    return result


def split_tetrahedra_by_linear_scalar_field(
    points,
    tetrahedra,
    scalar_field,
    *,
    global_vertex_ids=None,
    material_ids=None,
    cell_mass=None,
    retained_sides=(-1, 1),
):
    """Conformingly split a positively oriented tetrahedral mesh at a nodal linear field.

    The two closed half-domains are ``scalar_field <= 0`` (side -1) and
    ``scalar_field >= 0`` (side +1). A shared global-edge cache gives every
    incident tetrahedron the same intersection row. Each clipped boundary
    polygon is triangulated by a global-vertex-order pulling fan, and each
    convex clipped polytope is coned from its least ordered vertex. This makes
    the triangulation of a clipped shared triangle independent of which
    adjacent tetrahedron visited it.

    Original points are retained in source row order. New points are appended
    in sorted global-edge-key order and their rows of ``interpolation_weights``
    contain the exact linear edge interpolation weights. Original vertex order
    keys are ``(0, global_id)`` and cut keys are ``(1, min_id, max_id)``.
    Source material IDs are inherited unchanged; ``side_signs`` explicitly
    identifies each half for a caller's material policy. Both halves are
    always constructed and checked for positive Float64 cells and complete
    per-parent volume/mass partition before optional ``retained_sides`` filtering.
    The default ``(-1, +1)`` retains both halves. Per-cell mass, if supplied,
    is partitioned uniformly by each child's volume within its source parent.
    Tiny cuts are never silently dropped: malformed, zero-volume or returned
    Float32-collapsed output raises. Excluded sides remain accounted for in
    per-parent retained/excluded volume and mass arrays.
    """
    try:
        requested_sides = tuple(retained_sides)
    except TypeError as exc:
        raise ValueError("retained_sides must be a nonempty unique subset of (-1, +1)") from exc
    if (not requested_sides or any(isinstance(side, (bool, np.bool_)) or
            not isinstance(side, (int, np.integer)) for side in requested_sides) or
            any(int(side) not in (-1, 1) for side in requested_sides) or
            len({int(side) for side in requested_sides}) != len(requested_sides)):
        raise ValueError("retained_sides must be a nonempty unique subset of (-1, +1)")
    requested_sides = tuple(sorted(int(side) for side in requested_sides))
    P0 = _finite_points(points, "points")
    n = len(P0)
    if n == 0:
        raise ValueError("points must be nonempty")
    T0 = np.asarray(tetrahedra)
    if T0.ndim != 2 or T0.shape[1] != 4 or T0.dtype.kind not in "iu" or len(T0) == 0:
        raise ValueError("tetrahedra must be a nonempty integer Mx4 array")
    T0 = T0.astype(np.int64, copy=False)
    m = len(T0)
    if T0.min(initial=0) < 0 or T0.max(initial=0) >= n:
        raise ValueError("tetrahedra contain out-of-range point rows")
    if any(len(set(map(int, row))) != 4 for row in T0):
        raise ValueError("tetrahedra contain repeated vertex rows")
    source_keys = [tuple(sorted(map(int, row))) for row in T0]
    if len(set(source_keys)) != m:
        raise ValueError("duplicate source tetrahedra are not allowed")
    phi = np.asarray(scalar_field, dtype=np.float64)
    if phi.ndim != 1 or phi.shape != (n,) or not np.isfinite(phi).all():
        raise ValueError(f"scalar_field must be finite with shape ({n},)")
    if global_vertex_ids is None:
        gids = np.arange(n, dtype=np.int64)
    else:
        gids0 = np.asarray(global_vertex_ids)
        if gids0.ndim != 1 or gids0.shape != (n,) or gids0.dtype.kind not in "iu":
            raise ValueError(f"global_vertex_ids must be an integer vector of shape ({n},)")
        gids = gids0.astype(np.int64, copy=False)
        if len(np.unique(gids)) != n:
            raise ValueError("global_vertex_ids must be unique")
    gid_to_row = {int(g): i for i, g in enumerate(gids)}
    if material_ids is None:
        mats = None
    else:
        mats0 = np.asarray(material_ids)
        if mats0.ndim != 1 or mats0.shape != (m,) or mats0.dtype.kind not in "iu":
            raise ValueError(f"material_ids must be an integer vector of shape ({m},)")
        mats = mats0.astype(np.int64, copy=False)
    if cell_mass is None:
        masses = None
    else:
        masses0 = np.asarray(cell_mass, dtype=np.float64)
        if masses0.ndim != 1 or masses0.shape != (m,) or not np.isfinite(masses0).all() or np.any(masses0 < 0.0):
            raise ValueError(f"cell_mass must be finite, nonnegative and have shape ({m},)")
        masses = masses0

    # Input orientation is part of the caller's reference-mesh contract.
    # Never repair a negative cell by silently changing its connectivity.
    source_T = T0.copy()
    parent_volumes = np.empty(m, dtype=np.float64)
    for ci in range(m):
        det = _tet_det(P0, source_T[ci])
        if not np.isfinite(det) or det == 0.0:
            raise ValueError(f"source tetrahedron {ci} has zero or nonfinite volume")
        if det < 0.0:
            raise ValueError(f"source tetrahedron {ci} has negative orientation")
        parent_volumes[ci] = det / 6.0

    # Build every strict-crossing edge once in stable global endpoint order.
    crossing_keys = set()
    edge_rows = set()
    for tet in source_T:
        for a, b in ((tet[0], tet[1]), (tet[0], tet[2]), (tet[0], tet[3]),
                     (tet[1], tet[2]), (tet[1], tet[3]), (tet[2], tet[3])):
            a, b = int(a), int(b)
            key = tuple(sorted((int(gids[a]), int(gids[b]))))
            edge_rows.add(key)
            if _strictly_opposite(float(phi[a]), float(phi[b])):
                crossing_keys.add(key)
    crossing_keys = sorted(crossing_keys)
    gid_to_row_local = gid_to_row
    out_points = [row.copy() for row in P0]
    order_keys = [(0, int(g)) for g in gids]
    crossing_row = {}
    W_rows = list(range(n))
    W_cols = list(range(n))
    W_data = [1.0] * n
    for key in crossing_keys:
        ga, gb = key
        a = gid_to_row_local[ga]
        b = gid_to_row_local[gb]
        va, vb = float(phi[a]), float(phi[b])
        if not _strictly_opposite(va, vb):
            raise RuntimeError("global crossing-edge key does not have strict opposite signs")
        # Scale before addition/subtraction to avoid overflow for large
        # magnitudes and preserve tiny opposite-sign intersections.
        magnitude_scale = max(abs(va), abs(vb))
        aa, bb = abs(va) / magnitude_scale, abs(vb) / magnitude_scale
        t = aa / (aa + bb)
        if not (0.0 < t < 1.0) or not np.isfinite(t):
            raise ValueError(f"edge {key} has a numerically collapsed scalar intersection")
        point = P0[a] + t * (P0[b] - P0[a])
        if not np.isfinite(point).all():
            raise ValueError(f"edge {key} produced a nonfinite intersection")
        row = len(out_points)
        crossing_row[key] = row
        out_points.append(point)
        order_keys.append((1, ga, gb))
        W_rows.extend((row, row))
        W_cols.extend((a, b))
        W_data.extend((1.0 - t, t))
    P = np.asarray(out_points, dtype=np.float64)
    nout = len(P)
    W = sparse.csr_matrix((np.asarray(W_data, dtype=np.float64),
                           (np.asarray(W_rows, dtype=np.int64), np.asarray(W_cols, dtype=np.int64))),
                          shape=(nout, n), dtype=np.float64)
    W.sum_duplicates()
    W.sort_indices()
    phi_out = np.asarray(W @ phi, dtype=np.float64).reshape(-1)
    # Edges whose endpoint lies on the field use that existing endpoint row.
    def intersection_row(a, b):
        a, b = int(a), int(b)
        if phi[a] == 0.0:
            return a
        if phi[b] == 0.0:
            return b
        key = tuple(sorted((int(gids[a]), int(gids[b]))))
        try:
            return crossing_row[key]
        except KeyError as exc:
            raise ValueError(f"missing global intersection for crossing edge {key}") from exc

    face_local_indices = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    out_cells = []
    out_parent = []
    out_side = []
    def append_cell(parent, side, tet):
        tet = [int(x) for x in tet]
        if len(set(tet)) != 4:
            raise ValueError(f"parent {parent} generated a tetrahedron with repeated vertices")
        det = _tet_det(P, tet)
        if not np.isfinite(det) or det == 0.0:
            raise ValueError(f"parent {parent} generated a zero or nonfinite tetrahedron")
        if det < 0.0:
            tet[2], tet[3] = tet[3], tet[2]
        out_cells.append(tet)
        out_parent.append(int(parent))
        out_side.append(int(side))

    def make_clipped_side(parent, tet, side):
        vals = phi[tet]
        if not np.any(vals * side > 0.0):
            return 0
        if np.all(vals * side >= 0.0):
            append_cell(parent, side, tet)
            return 1
        facets = []
        for local_face in face_local_indices:
            face = [int(tet[i]) for i in local_face]
            poly = _clip_oriented_triangle(face, phi, side, intersection_row)
            if len(poly) >= 3:
                area2 = float(np.linalg.norm(_polygon_area_vector(P, poly)))
                if not np.isfinite(area2) or area2 == 0.0:
                    raise ValueError(f"parent {parent} has a degenerate clipped boundary facet")
                facets.append(poly)
        cut_rows = [int(v) for v in tet if phi[int(v)] == 0.0]
        for a, b in ((tet[0], tet[1]), (tet[0], tet[2]), (tet[0], tet[3]),
                     (tet[1], tet[2]), (tet[1], tet[3]), (tet[2], tet[3])):
            if _strictly_opposite(float(phi[int(a)]), float(phi[int(b)])):
                cut_rows.append(intersection_row(a, b))
        cut_poly = _ordered_cut_polygon(cut_rows, P, phi, tet, side, order_keys)
        facets.append(cut_poly)
        # The clipped convex polytope boundary must be a closed 2-manifold.
        from collections import Counter
        boundary_edges = Counter()
        for poly in facets:
            for i, a in enumerate(poly):
                b = poly[(i + 1) % len(poly)]
                if a == b:
                    raise ValueError(f"parent {parent} has a zero-length clipped facet edge")
                boundary_edges[tuple(sorted((int(a), int(b))))] += 1
        bad_edges = [e for e, c in boundary_edges.items() if c != 2]
        if bad_edges:
            raise ValueError(f"parent {parent} clipped boundary is open/nonmanifold at edges {bad_edges[:8]}")
        apex = min(set(v for poly in facets for v in poly), key=lambda r: order_keys[r])
        made = 0
        for poly in facets:
            triangles = _pulling_polygon_triangles(poly, order_keys)
            for tri in triangles:
                tri_area = float(np.linalg.norm(_polygon_area_vector(P, tri)))
                if not np.isfinite(tri_area) or tri_area == 0.0:
                    raise ValueError(f"parent {parent} pulling triangulation made a degenerate facet triangle")
                if apex in tri:
                    continue
                append_cell(parent, side, (apex, *tri))
                made += 1
        if made == 0:
            raise ValueError(f"parent {parent} strict side {side} generated no volume tetrahedra")
        return made

    for ci, tet0 in enumerate(source_T):
        vals = phi[tet0]
        if np.all(vals == 0.0):
            raise ValueError(f"parent tetrahedron {ci} has an identically zero scalar field")
        if np.any(vals < 0.0) and np.any(vals > 0.0):
            make_clipped_side(ci, tet0, -1)
            make_clipped_side(ci, tet0, +1)
        elif np.any(vals > 0.0):
            make_clipped_side(ci, tet0, +1)
        else:
            make_clipped_side(ci, tet0, -1)

    # Validate the complete split before selecting a retained side. This keeps
    # geometric partition and material accounting independent of the caller's
    # output-domain selection.
    all_C = np.asarray(out_cells, dtype=np.int64).reshape(-1, 4)
    all_parent_rows = np.asarray(out_parent, dtype=np.int64)
    all_sides = np.asarray(out_side, dtype=np.int8)
    if len(all_C) == 0 or len(all_parent_rows) != len(all_C) or len(all_sides) != len(all_C):
        raise RuntimeError("split produced an empty or inconsistent cell ancestry table")
    if np.any((all_parent_rows < 0) | (all_parent_rows >= m)) or not np.isin(all_sides, (-1, 1)).all():
        raise RuntimeError("split produced invalid parent or side labels")
    # Every child must lie wholly in its declared closed half-domain. Allow
    # only scale-relative roundoff at interpolated cut vertices.
    field_tol = 256.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(phi), initial=0.0)), np.finfo(np.float64).tiny)
    signed_child_values = phi_out[all_C] * all_sides[:, None]
    if np.any(signed_child_values < -field_tol):
        bad = np.argwhere(signed_child_values < -field_tol)[:16].tolist()
        raise ValueError(f"generated child lies across its declared scalar half-domain at {bad}")
    if len({tuple(sorted(map(int, t))) for t in all_C}) != len(all_C):
        raise ValueError("split produced duplicate tetrahedra")
    full_used = set(map(int, all_C.ravel()))
    unused_intersections_before_filter = [r for r in range(n, nout) if r not in full_used]
    if unused_intersections_before_filter:
        raise ValueError(f"full split produced unused intersection vertices {unused_intersections_before_filter[:16]}")

    all_det64 = np.asarray([_tet_det(P, tet) for tet in all_C], dtype=np.float64)
    if np.any(~np.isfinite(all_det64) | (all_det64 <= 0.0)):
        bad = np.flatnonzero(~np.isfinite(all_det64) | (all_det64 <= 0.0))[:16].tolist()
        raise ValueError(f"full split contains nonpositive Float64 children at rows {bad}")
    all_child_volumes = all_det64 / 6.0
    all_parent_sums = np.bincount(all_parent_rows, weights=all_child_volumes, minlength=m)
    parent_abs_error = np.abs(all_parent_sums - parent_volumes)
    parent_rel_error = parent_abs_error / parent_volumes
    volume_tolerance = 4096.0 * np.finfo(np.float64).eps * parent_volumes
    bad_parent = np.flatnonzero(parent_abs_error > volume_tolerance)
    if bad_parent.size:
        raise ValueError(f"full child volumes do not conserve parent volumes at rows {bad_parent[:16].tolist()}")

    all_materials = None if mats is None else mats[all_parent_rows]
    all_child_mass = None
    max_parent_mass_error = None
    if masses is not None:
        all_child_mass = masses[all_parent_rows] * (all_child_volumes / parent_volumes[all_parent_rows])
        mass_sums = np.bincount(all_parent_rows, weights=all_child_mass, minlength=m)
        mass_error = np.abs(mass_sums - masses)
        mass_tol = 8192.0 * np.finfo(np.float64).eps * np.maximum(np.abs(masses), np.finfo(np.float64).tiny)
        if np.any(mass_error > mass_tol):
            bad = np.flatnonzero(mass_error > mass_tol)
            raise ValueError(f"full child mass does not partition source parent masses at rows {bad[:16].tolist()}")
        max_parent_mass_error = float(np.max(mass_error, initial=0.0))

    selected_mask = np.isin(all_sides, np.asarray(requested_sides, dtype=np.int8))
    if not np.any(selected_mask):
        raise ValueError(f"retained_sides={requested_sides} selects no output cells")
    excluded_mask = ~selected_mask
    retained_parent_volumes = np.bincount(
        all_parent_rows[selected_mask], weights=all_child_volumes[selected_mask], minlength=m)
    excluded_parent_volumes = np.bincount(
        all_parent_rows[excluded_mask], weights=all_child_volumes[excluded_mask], minlength=m)
    partition_error = np.abs(retained_parent_volumes + excluded_parent_volumes - parent_volumes)
    if np.any(partition_error > volume_tolerance):
        bad = np.flatnonzero(partition_error > volume_tolerance)
        raise ValueError(f"retained/excluded parent volumes do not partition source at rows {bad[:16].tolist()}")

    retained_parent_masses = None
    excluded_parent_masses = None
    if masses is not None:
        retained_parent_masses = np.bincount(
            all_parent_rows[selected_mask], weights=all_child_mass[selected_mask], minlength=m)
        excluded_parent_masses = np.bincount(
            all_parent_rows[excluded_mask], weights=all_child_mass[excluded_mask], minlength=m)
        mass_partition_error = np.abs(retained_parent_masses + excluded_parent_masses - masses)
        mass_tol = 8192.0 * np.finfo(np.float64).eps * np.maximum(np.abs(masses), np.finfo(np.float64).tiny)
        if np.any(mass_partition_error > mass_tol):
            bad = np.flatnonzero(mass_partition_error > mass_tol)
            raise ValueError(f"retained/excluded parent masses do not partition source at rows {bad[:16].tolist()}")

    C = all_C[selected_mask]
    parent_rows = all_parent_rows[selected_mask]
    sides = all_sides[selected_mask]
    child_volumes = all_child_volumes[selected_mask]
    materials_out = None if all_materials is None else all_materials[selected_mask]
    child_mass = None if all_child_mass is None else all_child_mass[selected_mask]
    det64 = all_det64[selected_mask]
    used_after_filter = set(map(int, C.ravel()))
    unused_intersections_after_filter = [r for r in range(n, nout) if r not in used_after_filter]

    P32 = P.astype(np.float32)
    det32 = np.empty(len(C), dtype=np.float64)
    q32 = np.empty(len(C), dtype=np.float64)
    for i, tet in enumerate(C):
        p = P32[tet].astype(np.float64)
        det = float(np.dot(p[1]-p[0], np.cross(p[2]-p[0], p[3]-p[0])))
        edge_lengths = [float(np.linalg.norm(p[a] - p[b]))
                        for a, b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))]
        longest = max(edge_lengths, default=0.0)
        det32[i] = det
        q32[i] = det / (longest ** 3) if longest > 0.0 else float("nan")
    bad32 = np.flatnonzero(~np.isfinite(det32) | (det32 <= 0.0) | ~np.isfinite(q32))
    if bad32.size:
        raise ValueError(f"Float32 emission contains nonpositive or invalid tetrahedra at rows {bad32[:16].tolist()}")
    quality_warning_rows = np.flatnonzero(q32 < SPLIT_QUALITY_WARNING_THRESHOLD)
    quality_warning_parent_rows = np.unique(parent_rows[quality_warning_rows])
    det64 = np.asarray([_tet_det(P, tet) for tet in C], dtype=np.float64)
    if np.any(~np.isfinite(det64) | (det64 <= 0.0)):
        raise RuntimeError("positive Float64 cell orientation invariant failed")

    material_volume_before = {}
    material_volume_after = {}
    if mats is not None:
        for mid in np.unique(mats):
            material_volume_before[str(int(mid))] = float(parent_volumes[mats == mid].sum())
            material_volume_after[str(int(mid))] = float(child_volumes[materials_out == mid].sum())
    diagnostics = {
        "status": "STATIC_CONFORMING_SCALAR_SPLIT",
        "physical_or_native_admission": False,
        "source_vertex_count": int(n),
        "output_vertex_count": int(nout),
        "source_tetrahedron_count": int(m),
        "output_tetrahedron_count": int(len(C)),
        "prefilter_child_count": int(len(all_C)),
        "selected_sides": list(requested_sides),
        "excluded_child_count": int(np.count_nonzero(excluded_mask)),
        "split_source_tetrahedron_count": int(np.count_nonzero(np.bincount(parent_rows, minlength=m) > 1)),
        "intersection_vertex_count": int(len(crossing_keys)),
        "intersection_edges_global_ids": [list(map(int, k)) for k in crossing_keys],
        "all_source_cells_have_children": bool(np.all(np.bincount(parent_rows, minlength=m) > 0)),
        "all_source_cells_have_children_before_filter": bool(np.all(np.bincount(all_parent_rows, minlength=m) > 0)),
        "side_negative_cell_count": int(np.count_nonzero(sides < 0)),
        "side_positive_cell_count": int(np.count_nonzero(sides > 0)),
        "parent_volume_max_abs_error": float(np.max(parent_abs_error, initial=0.0)),
        "parent_volume_max_relative_error": float(np.max(parent_rel_error, initial=0.0)),
        "prefilter_parent_volume_max_abs_error": float(np.max(parent_abs_error, initial=0.0)),
        "prefilter_parent_volume_max_relative_error": float(np.max(parent_rel_error, initial=0.0)),
        "retained_parent_volume_total": float(np.sum(retained_parent_volumes)),
        "excluded_parent_volume_total": float(np.sum(excluded_parent_volumes)),
        "retained_excluded_parent_volume_partition_max_abs_error": float(np.max(partition_error, initial=0.0)),
        "parent_volume_tolerance_relative": float(4096.0 * np.finfo(np.float64).eps),
        "per_material_parent_volume_before": material_volume_before,
        "per_material_child_volume_after": material_volume_after,
        "per_material_excluded_volume": {} if mats is None else {
            str(int(mid)): float(np.sum(excluded_parent_volumes[mats == mid])) for mid in np.unique(mats)},
        "parent_mass_max_abs_partition_error": max_parent_mass_error,
        "prefilter_parent_mass_max_abs_partition_error": max_parent_mass_error,
        "retained_parent_mass_total": None if retained_parent_masses is None else float(np.sum(retained_parent_masses)),
        "excluded_parent_mass_total": None if excluded_parent_masses is None else float(np.sum(excluded_parent_masses)),
        "retained_excluded_parent_mass_partition_max_abs_error": None if retained_parent_masses is None else float(np.max(np.abs(retained_parent_masses + excluded_parent_masses - masses), initial=0.0)),
        "unused_intersection_vertex_count_before_filter": int(len(unused_intersections_before_filter)),
        "unused_intersection_vertex_count_after_filter": int(len(unused_intersections_after_filter)),
        "unused_intersection_vertex_rows_after_filter_first16": unused_intersections_after_filter[:16],
        "interpolation_weight_row_sum_error_max": float(np.max(np.abs(np.asarray(W.sum(axis=1)).ravel() - 1.0), initial=0.0)),
        "interpolated_scalar_max_abs_residual_at_cut_rows": float(np.max(np.abs(phi_out[n:]), initial=0.0)),
        "float64_min_determinant": float(np.min(det64)),
        "float32_min_determinant": float(np.min(det32)),
        "float32_nonpositive_cell_count": int(len(bad32)),
        "float32_min_owner_quality_det_over_longest_edge_cubed": float(np.min(q32)),
        "float32_quality_warning_threshold_det_over_longest_edge_cubed": SPLIT_QUALITY_WARNING_THRESHOLD,
        "float32_quality_warning": bool(quality_warning_rows.size),
        "float32_quality_warning_cell_count": int(quality_warning_rows.size),
        "float32_quality_warning_parent_count": int(quality_warning_parent_rows.size),
        "float32_quality_warning_cell_rows_first16": quality_warning_rows[:16].astype(int).tolist(),
        "float32_quality_warning_parent_rows_first16": quality_warning_parent_rows[:16].astype(int).tolist(),
        "float32_minimum_quality_is_diagnostic_not_admission": True,
        "source_material_ids_preserved": bool(mats is None or np.array_equal(materials_out, mats[parent_rows])),
        "source_vertex_positions_preserved": bool(np.array_equal(P[:n], P0)),
        "interpolation_weights_max_nonzeros_per_row": int(max(np.diff(W.indptr), default=0)),
    }
    return ScalarFieldVolumeSplit(
        points=P, points_float32=P32, tetrahedra=C,
        parent_cell_rows=parent_rows, side_signs=sides,
        material_ids=materials_out, cell_masses=child_mass,
        interpolation_weights=W, scalar_values=phi_out,
        global_vertex_order_keys=tuple(order_keys),
        intersection_edge_global_ids=np.asarray(crossing_keys, dtype=np.int64).reshape(-1, 2),
        retained_parent_volumes=retained_parent_volumes,
        excluded_parent_volumes=excluded_parent_volumes,
        retained_parent_masses=retained_parent_masses,
        excluded_parent_masses=excluded_parent_masses,
        diagnostics=diagnostics,
    )


def snap_shared_scalar_interface_vertices(points, tets, values, hskin, *, fraction):
    """Set one globally shared vertex to exact zero only if every incident crossing-edge
    intersection is within a declared fraction of that vertex's local skin depth.
    Coordinates remain unchanged; the explicit edge-interface displacement is retained
    in the ledger. This opt-in static construction perturbation neither certifies
    clearance nor relaxes the downstream splitter positivity/partition gates.
    """
    P = _finite_points(points, "points")
    raw_tets = np.asarray(tets)
    if raw_tets.ndim != 2 or raw_tets.shape[1] != 4 or raw_tets.dtype.kind not in "iu":
        raise ValueError("tets must be an integer (N, 4) array")
    T = raw_tets.astype(np.int64, copy=False)
    if not len(T) or T.min() < 0 or T.max() >= len(P):
        raise ValueError("empty or out-of-range tets")
    if np.any(np.diff(np.sort(T, axis=1), axis=1) == 0):
        raise ValueError("repeated tetrahedron vertex")
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction < 0.5:
        raise ValueError("fraction must be explicit, finite and in [0, 0.5)")
    phi = np.asarray(values, dtype=np.float64)
    hs = _positive_depth_field(hskin, "hskin", len(P))
    if phi.shape != (len(P),) or not np.isfinite(phi).all():
        raise ValueError("values must be a finite per-vertex scalar field")
    phi = phi.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        P32 = P.astype(np.float32).astype(np.float64)
    if not np.isfinite(P32).all():
        raise ValueError("points must remain finite in emitted Float32")
    X = P32[T]
    det = np.einsum("ij,ij->i", X[:,1]-X[:,0], np.cross(X[:,2]-X[:,0],X[:,3]-X[:,0]))
    if not np.isfinite(det).all() or np.any(det <= 0):
        raise ValueError("input cells must be positive in emitted Float32")
    if np.any(np.all(phi[T] == 0.0, axis=1)):
        raise ValueError("ambiguous all-zero input parent")
    raw=np.concatenate((T[:,[0,1]],T[:,[0,2]],T[:,[0,3]],T[:,[1,2]],T[:,[1,3]],T[:,[2,3]]),axis=0)
    E=np.unique(np.sort(raw,axis=1),axis=0)
    a,b=E.T
    va=phi[a];vb=phi[b]
    cross=((va<0)&(vb>0))|((va>0)&(vb<0))
    e=E[cross];a=e[:,0];b=e[:,1];va=phi[a];vb=phi[b]
    if len(e)==0:
        return phi,{"edge_count":0,"snapped_vertex_count":0},None
    den=np.abs(va)+np.abs(vb)
    t=np.abs(va)/den
    d=P[b]-P[a];L=np.linalg.norm(d,axis=1)
    cut=P[a]+t[:,None]*d
    cf=cut.astype(np.float32).astype(np.float64)
    cf32=cf.astype(np.float32)
    prev=np.nextafter(cf32,np.full_like(cf32,-np.inf,dtype=np.float32)).astype(np.float64)
    nxt=np.nextafter(cf32,np.full_like(cf32,np.inf,dtype=np.float32)).astype(np.float64)
    round_bound=np.linalg.norm(0.5*np.maximum(np.abs(cf-prev),np.abs(nxt-cf)),axis=1)
    da=t*L;db=(1-t)*L
    maxdist=np.zeros(len(P),dtype=np.float64);nedge=np.zeros(len(P),dtype=np.int32)
    np.maximum.at(maxdist,a,da);np.maximum.at(maxdist,b,db)
    np.add.at(nedge,a,1);np.add.at(nedge,b,1)
    cap=fraction*hs
    snapped=(nedge>0)&(maxdist<=cap)
    out=phi.copy();out[snapped]=0.0
    if np.any(np.all(out[T] == 0.0, axis=1)):
        raise ValueError("snap would create an ambiguous all-zero parent")
    endpoint_move=np.column_stack((np.where(snapped[a],da,0.0),np.where(snapped[b],db,0.0)))
    snapped_endpoint_ratios=np.concatenate((round_bound[snapped[a]]/da[snapped[a]],round_bound[snapped[b]]/db[snapped[b]]))
    stats={"edge_count":int(len(e)),"snapped_vertex_count":int(snapped.sum()),
      "snap_fraction_of_local_hskin":float(fraction),"max_projected_interface_displacement_m":float(maxdist[snapped].max(initial=0)),
      "max_allowed_displacement_m":float(cap[snapped].max(initial=0)),
      "max_cutpoint_float32_rounding_bound_m":float(round_bound.max(initial=0)),
      "max_cutpoint_rounding_to_snap_displacement_ratio":float(snapped_endpoint_ratios.max(initial=0)) if len(snapped_endpoint_ratios) else None}
    ledger={"original_phi_m":phi,"snapped_phi_m":out,"snap_delta_m":out-phi,
      "local_hskin_m":hs,"allowed_displacement_m":cap,"max_incident_crossing_displacement_m":maxdist,
      "incident_crossing_edge_count":nedge,"snapped_vertex_rows":np.flatnonzero(snapped).astype(np.int64),
      "crossing_edges":e.astype(np.int64),"crossing_parameter_from_a":t,"crossing_edge_length_m":L,
      "crossing_distance_to_a_m":da,"crossing_distance_to_b_m":db,
      "crossing_cutpoint_f32_rounding_bound_m":round_bound,"crossing_snapped_a":snapped[a],
      "crossing_snapped_b":snapped[b],"crossing_projected_interface_displacement_a_m":endpoint_move[:,0],
      "crossing_projected_interface_displacement_b_m":endpoint_move[:,1]}
    return out,stats,ledger
