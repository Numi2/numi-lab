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

