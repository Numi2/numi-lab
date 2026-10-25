#!/usr/bin/env python3
"""Bounded same-immediate-parent fixed-coordinate surface flips (349).

This module proposes only topology changes. It does not run self-intersection,
bone, mechanics, or native audits and never moves point coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

FACE_ROW_KEYS = {
    "source_O_faces",
    "area_reference_037_source_O_faces",
    "face_kind",
    "parent_coarse_face_row",
    "parent_fine_detail_face_row",
    "parent_source_face_marker",
    "parent_fine_subface_row",
    "refinement_face_parent_face_row",
    "refinement_face_parent_barycentric",
}
CSR_VALUE_KEYS = {"face_source_ancestry_face_rows", "face_fine_subface_ancestry_rows"}
METADATA_KEYS = (
    "parent_coarse_face_row",
    "parent_fine_detail_face_row",
    "parent_source_face_marker",
    "parent_fine_subface_row",
)
FLOAT_EPS = np.finfo(np.float64).eps


class FlipInputError(ValueError):
    pass


def _edge_map(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    result: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for fi, tri in enumerate(faces):
        for a, b in zip(tri, np.roll(tri, -1)):
            aa, bb = int(a), int(b)
            result.setdefault((min(aa, bb), max(aa, bb)), []).append((fi, aa, bb))
    return result


def _csr_validate(ptr: np.ndarray, values: np.ndarray, rows: int, name: str) -> None:
    if ptr.ndim != 1 or ptr.shape != (rows + 1,):
        raise FlipInputError(f"{name} indptr has wrong shape")
    if values.ndim != 1 or int(ptr[0]) != 0 or int(ptr[-1]) != len(values):
        raise FlipInputError(f"{name} CSR endpoints are invalid")
    if np.any(ptr[1:] < ptr[:-1]):
        raise FlipInputError(f"{name} CSR indptr is not monotone")


def _csr_get(ptr: np.ndarray, values: np.ndarray, row: int) -> list[int]:
    return [int(x) for x in values[int(ptr[row]):int(ptr[row + 1])]]


def _area_vector(points: np.ndarray, tris: Sequence[Sequence[int]]) -> np.ndarray:
    total = np.zeros(3, dtype=np.float64)
    for tri in tris:
        p = points[np.asarray(tri, dtype=np.int64)].astype(np.float64, copy=False)
        total += 0.5 * np.cross(p[1] - p[0], p[2] - p[0])
    return total


def _normal(points: np.ndarray, tri: Sequence[int]) -> tuple[np.ndarray, float]:
    p = points[np.asarray(tri, dtype=np.int64)].astype(np.float64, copy=False)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    length = float(np.linalg.norm(n))
    return (n / length if length > 0.0 else np.zeros(3)), 0.5 * length


def _min_angle_deg(points: np.ndarray, tri: Sequence[int]) -> float:
    p = points[np.asarray(tri, dtype=np.int64)].astype(np.float64, copy=False)
    a, b, c = p
    e0, e1, e2 = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)
    if min(e0, e1, e2) <= 0.0:
        return 0.0
    c0 = np.clip(np.dot(b - a, c - a) / (e0 * e2), -1.0, 1.0)
    c1 = np.clip(np.dot(a - b, c - b) / (e0 * e1), -1.0, 1.0)
    a0, a1 = np.arccos(c0), np.arccos(c1)
    return float(np.rad2deg(min(a0, a1, np.pi - a0 - a1)))


def _chart(points: np.ndarray, normal: np.ndarray, rows: Sequence[int]) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64)
    nlen = float(np.linalg.norm(n))
    if not np.isfinite(nlen) or nlen <= 64.0 * FLOAT_EPS:
        raise FlipInputError("cannot form a stable local chart")
    n = n / nlen
    axis = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = np.cross(n, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    p = points[np.asarray(rows, dtype=np.int64)].astype(np.float64, copy=False)
    return np.column_stack((p @ e1, p @ e2))


def _strict_convex(poly: np.ndarray) -> bool:
    if poly.shape != (4, 2) or not np.isfinite(poly).all():
        return False
    edges = np.roll(poly, -1, axis=0) - poly
    scale2 = float(np.max(np.sum(edges * edges, axis=1)))
    if not np.isfinite(scale2) or scale2 <= 0.0:
        return False
    crosses = np.asarray([
        edges[i, 0] * edges[(i + 1) % 4, 1] -
        edges[i, 1] * edges[(i + 1) % 4, 0]
        for i in range(4)
    ])
    tol = 128.0 * FLOAT_EPS * scale2
    return bool(np.all(crosses > tol) or np.all(crosses < -tol))


def _array_sha(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def _validate_input(data: Mapping[str, np.ndarray]) -> None:
    required = (
        "points_world_m",
        "source_O_faces",
        "area_reference_037_points_world_m_f64",
        "area_reference_037_source_O_faces",
        "refinement_face_parent_face_row",
        "refinement_face_parent_barycentric",
        "face_source_ancestry_indptr",
        "face_source_ancestry_face_rows",
        "face_fine_subface_ancestry_indptr",
        "face_fine_subface_ancestry_rows",
        "face_kind",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise FlipInputError(f"missing required candidate arrays: {missing}")
    x = np.asarray(data["points_world_m"])
    f = np.asarray(data["source_O_faces"])
    rf = np.asarray(data["area_reference_037_source_O_faces"])
    r = np.asarray(data["area_reference_037_points_world_m_f64"])
    parent = np.asarray(data["refinement_face_parent_face_row"])
    bary = np.asarray(data["refinement_face_parent_barycentric"])
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise FlipInputError("points_world_m must be finite Nx3")
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise FlipInputError("source_O_faces must be integer Fx3")
    if f.size and (int(f.min()) < 0 or int(f.max()) >= len(x)):
        raise FlipInputError("source face node row is out of range")
    if rf.shape != f.shape or not np.array_equal(rf, f):
        raise FlipInputError("matched-037 reference faces must exactly match current face connectivity")
    if r.shape != x.shape or not np.isfinite(r).all():
        raise FlipInputError("matched-037 points must be finite Nx3")
    if parent.shape != (len(f),) or not np.issubdtype(parent.dtype, np.integer):
        raise FlipInputError("refinement parent rows must be integer F")
    if bary.shape != (len(f), 3, 3) or not np.isfinite(bary).all():
        raise FlipInputError("face barycentric ancestry must be finite Fx3x3")
    if np.any(parent < 0):
        raise FlipInputError("negative/missing refinement face parent row")
    if np.max(np.abs(np.sum(bary, axis=2) - 1.0), initial=0.0) > 1.0e-10:
        raise FlipInputError("parent barycentric rows do not sum to one")
    if np.min(bary, initial=0.0) < -1.0e-10 or np.max(bary, initial=1.0) > 1.0 + 1.0e-10:
        raise FlipInputError("parent barycentric coordinate outside simplex")
    if np.asarray(data["face_kind"]).shape != (len(f),):
        raise FlipInputError("face_kind must be face aligned")
    for key in METADATA_KEYS:
        if key in data:
            meta = np.asarray(data[key])
            if meta.shape != (len(f),):
                raise FlipInputError(f"{key} must be face aligned")
            if not np.issubdtype(meta.dtype, np.signedinteger):
                raise FlipInputError(f"{key} must be signed integer so ambiguity can be marked -1")
    for key, value in data.items():
        a = np.asarray(value)
        if a.ndim and a.shape[0] == len(f) and key not in FACE_ROW_KEYS and key not in CSR_VALUE_KEYS:
            raise FlipInputError(f"unclassified face-row array requires explicit update policy: {key}")
    _csr_validate(np.asarray(data["face_source_ancestry_indptr"]),
                  np.asarray(data["face_source_ancestry_face_rows"]), len(f), "face source ancestry")
    _csr_validate(np.asarray(data["face_fine_subface_ancestry_indptr"]),
                  np.asarray(data["face_fine_subface_ancestry_rows"]), len(f), "fine subface ancestry")


def _pair_candidate(
    data: Mapping[str, np.ndarray],
    edge_map: Mapping[tuple[int, int], list[tuple[int, int, int]]],
    incidences: list[tuple[int, int, int]],
    seeds: set[int],
    min_angle_gain_deg: float,
    min_normal_dot: float,
) -> tuple[dict | None, str]:
    f = np.asarray(data["source_O_faces"])
    x = np.asarray(data["points_world_m"])
    r = np.asarray(data["area_reference_037_points_world_m_f64"])
    parent = np.asarray(data["refinement_face_parent_face_row"])
    bary = np.asarray(data["refinement_face_parent_barycentric"])
    if len(incidences) != 2:
        return None, "not_interior_edge"
    i, u, v = incidences[0]
    j, u2, v2 = incidences[1]
    if i not in seeds and j not in seeds:
        return None, "outside_seed_faces"
    if parent[i] != parent[j] or parent[i] < 0:
        return None, "parent_mismatch"
    if (u2, v2) != (v, u):
        return None, "directed_winding_mismatch"
    shared = {u, v}
    a = next(int(row) for row in f[i] if int(row) not in shared)
    b = next(int(row) for row in f[j] if int(row) not in shared)
    if len({u, v, a, b}) != 4:
        return None, "non_four_vertex_patch"
    if (min(a, b), max(a, b)) in edge_map:
        return None, "alternate_diagonal_already_exists"

    bary_by_node: dict[int, np.ndarray] = {}
    for fi in (i, j):
        for k, node in enumerate(f[fi]):
            node = int(node)
            row = np.asarray(bary[fi, k], dtype=np.float64)
            previous = bary_by_node.get(node)
            if previous is not None and np.max(np.abs(previous - row)) > 1.0e-10:
                return None, "inconsistent_shared_vertex_parent_barycentric"
            bary_by_node[node] = row
    # Boundary order of the oriented two-face patch is v,a,u,b.
    polygon_rows = [v, a, u, b]
    bary_poly = np.asarray([bary_by_node[node][:2] for node in polygon_rows])
    if not _strict_convex(bary_poly):
        return None, "parent_patch_nonconvex_or_degenerate"
    # Certify that the four stored 037 vertices really belong to this one
    # barycentric parent triangle, instead of trusting row labels alone.
    parent_weights = np.asarray([bary_by_node[node] for node in polygon_rows])
    parent_reference_vertices, _, rank, _ = np.linalg.lstsq(
        parent_weights, r[np.asarray(polygon_rows, dtype=np.int64)], rcond=None
    )
    if rank != 3:
        return None, "reference_parent_barycentric_rank_deficient"
    ref_reconstruction = parent_weights @ parent_reference_vertices
    ref_residual = np.linalg.norm(
        ref_reconstruction - r[np.asarray(polygon_rows, dtype=np.int64)], axis=1
    )
    coordinate_scale = max(1.0, float(np.max(np.abs(r[np.asarray(polygon_rows, dtype=np.int64)]))))
    ref_residual_bound = max(1.0e-12, 256.0 * FLOAT_EPS * coordinate_scale)
    if float(ref_residual.max(initial=0.0)) > ref_residual_bound:
        return None, "reference_parent_barycentric_reconstruction_mismatch"
    old_cur_normals = [_normal(x, f[fi])[0] for fi in (i, j)]
    old_ref_normals = [_normal(r, f[fi])[0] for fi in (i, j)]
    if min(np.linalg.norm(n) for n in old_cur_normals + old_ref_normals) <= 0.0:
        return None, "old_face_degenerate"
    cur_avg = old_cur_normals[0] + old_cur_normals[1]
    ref_avg = old_ref_normals[0] + old_ref_normals[1]
    if np.linalg.norm(cur_avg) <= 64.0 * FLOAT_EPS or np.linalg.norm(ref_avg) <= 64.0 * FLOAT_EPS:
        return None, "parent_patch_orientation_ambiguous"
    cur_avg /= np.linalg.norm(cur_avg)
    ref_avg /= np.linalg.norm(ref_avg)
    if min(float(np.dot(n, cur_avg)) for n in old_cur_normals) <= min_normal_dot:
        return None, "current_old_faces_folded"
    if min(float(np.dot(n, ref_avg)) for n in old_ref_normals) <= min_normal_dot:
        return None, "reference_old_faces_folded"
    if min(float(np.dot(nc, nr)) for nc, nr in zip(old_cur_normals, old_ref_normals)) <= min_normal_dot:
        return None, "current_reference_orientation_disagreement"
    try:
        if not _strict_convex(_chart(x, cur_avg, polygon_rows)):
            return None, "current_patch_nonconvex_or_degenerate"
    except FlipInputError:
        return None, "current_patch_chart_unstable"

    new_faces = (
        np.asarray([u, b, a], dtype=f.dtype),
        np.asarray([b, v, a], dtype=f.dtype),
    )
    old_current_angles = [_min_angle_deg(x, f[fi]) for fi in (i, j)]
    new_current_normals = [_normal(x, tri)[0] for tri in new_faces]
    new_reference_normals = [_normal(r, tri)[0] for tri in new_faces]
    new_current_areas = [_normal(x, tri)[1] for tri in new_faces]
    new_reference_areas = [_normal(r, tri)[1] for tri in new_faces]
    current_scale2 = max(
        float(np.sum((x[p] - x[q]) ** 2))
        for p, q in ((u, v), (v, a), (a, u), (u, b), (b, v))
    )
    reference_scale2 = max(
        float(np.sum((r[p] - r[q]) ** 2))
        for p, q in ((u, v), (v, a), (a, u), (u, b), (b, v))
    )
    current_floor = 128.0 * FLOAT_EPS * current_scale2
    reference_floor = 128.0 * FLOAT_EPS * reference_scale2
    if min(new_current_areas) <= current_floor:
        return None, "current_replacement_face_degenerate"
    if min(new_reference_areas) <= reference_floor:
        return None, "reference_replacement_face_degenerate"
    if min(float(np.dot(n, cur_avg)) for n in new_current_normals) <= min_normal_dot:
        return None, "current_replacement_orientation_mismatch"
    if min(float(np.dot(n, ref_avg)) for n in new_reference_normals) <= min_normal_dot:
        return None, "reference_replacement_orientation_mismatch"
    if min(float(np.dot(nc, nr)) for nc, nr in zip(new_current_normals, new_reference_normals)) <= min_normal_dot:
        return None, "current_reference_replacement_orientation_disagreement"
    old_current_area_vector = _area_vector(x, [f[i], f[j]])
    new_current_area_vector = _area_vector(x, new_faces)
    old_reference_area_vector = _area_vector(r, [f[i], f[j]])
    new_reference_area_vector = _area_vector(r, new_faces)
    current_area_vector_delta = float(np.linalg.norm(new_current_area_vector - old_current_area_vector))
    reference_area_vector_delta = float(np.linalg.norm(new_reference_area_vector - old_reference_area_vector))
    if current_area_vector_delta > 1024.0 * FLOAT_EPS * current_scale2:
        return None, "current_parent_area_vector_changed"
    if reference_area_vector_delta > 1024.0 * FLOAT_EPS * reference_scale2:
        return None, "reference_parent_area_vector_changed"
    old_pair_angle = min(old_current_angles)
    new_pair_angle = min(_min_angle_deg(x, tri) for tri in new_faces)
    gain = new_pair_angle - old_pair_angle
    if gain <= min_angle_gain_deg:
        return None, "minimum_angle_not_improved"

    source_rows = [
        _csr_get(np.asarray(data["face_source_ancestry_indptr"]),
                 np.asarray(data["face_source_ancestry_face_rows"]), fi)
        for fi in (i, j)
    ]
    subface_rows = [
        _csr_get(np.asarray(data["face_fine_subface_ancestry_indptr"]),
                 np.asarray(data["face_fine_subface_ancestry_rows"]), fi)
        for fi in (i, j)
    ]
    if any(not rows for rows in source_rows) or any(not rows for rows in subface_rows):
        return None, "missing_face_ancestry"
    if int(np.asarray(data["face_kind"])[i]) != int(np.asarray(data["face_kind"])[j]):
        return None, "face_kind_mismatch"
    source_union = sorted(set(source_rows[0]) | set(source_rows[1]))
    subface_union = sorted(set(subface_rows[0]) | set(subface_rows[1]))
    return {
        "face_rows": [int(i), int(j)],
        "parent_row": int(parent[i]),
        "seed_face_rows": sorted(set([i, j]) & seeds),
        "edge": [int(min(u, v)), int(max(u, v))],
        "new_diagonal": [int(min(a, b)), int(max(a, b))],
        "new_faces": [new_faces[0].tolist(), new_faces[1].tolist()],
        "old_min_angle_deg": float(old_pair_angle),
        "new_min_angle_deg": float(new_pair_angle),
        "angle_gain_deg": float(gain),
        "old_current_areas_m2": [float(_normal(x, f[fi])[1]) for fi in (i, j)],
        "new_current_areas_m2": [float(vv) for vv in new_current_areas],
        "old_reference_areas_m2": [float(_normal(r, f[fi])[1]) for fi in (i, j)],
        "new_reference_areas_m2": [float(vv) for vv in new_reference_areas],
        "current_parent_area_vector_delta_m2": current_area_vector_delta,
        "reference_parent_area_vector_delta_m2": reference_area_vector_delta,
        "source_face_ancestry_union": source_union,
        "fine_subface_ancestry_union": subface_union,
        "barycentric_by_vertex": {str(node): bary_by_node[node].tolist()
                                  for node in sorted(bary_by_node)},
    }, "candidate"


def _rebuild_csr(ptr: np.ndarray, values: np.ndarray, updates: Mapping[int, list[int]], rows: int):
    new_ptr = np.zeros(rows + 1, dtype=np.int64)
    new_values: list[int] = []
    for row in range(rows):
        new_values.extend(updates.get(row, _csr_get(ptr, values, row)))
        new_ptr[row + 1] = len(new_values)
    return new_ptr, np.asarray(new_values, dtype=values.dtype)


def _edge_stats(faces: np.ndarray):
    em = _edge_map(faces)
    boundary = {edge for edge, inc in em.items() if len(inc) == 1}
    overfull = sorted((edge, len(inc)) for edge, inc in em.items() if len(inc) > 2)
    winding_conflicts = sum(
        1 for inc in em.values()
        if len(inc) == 2 and (inc[0][1], inc[0][2]) == (inc[1][1], inc[1][2])
    )
    return em, boundary, overfull, int(winding_conflicts)




def _parent_rings(faces: np.ndarray, parents: np.ndarray):
    per_parent: dict[int, dict[tuple[int, int], list[tuple[int, int]]]] = {}
    for fi, tri in enumerate(faces):
        parent = int(parents[fi])
        bucket = per_parent.setdefault(parent, {})
        for a, b in zip(tri, np.roll(tri, -1)):
            aa, bb = int(a), int(b)
            bucket.setdefault((min(aa, bb), max(aa, bb)), []).append((aa, bb))
    rings = {}
    overfull = []
    for parent, edge_rows in per_parent.items():
        boundary = []
        for edge, directed in edge_rows.items():
            if len(directed) == 1:
                boundary.append(directed[0])
            elif len(directed) > 2:
                overfull.append((parent, edge, len(directed)))
        rings[parent] = tuple(sorted(boundary))
    return rings, sorted(overfull)

def propose_flip_batch(
    candidate: Mapping[str, np.ndarray],
    target_face_rows: Sequence[int],
    *,
    max_flips: int = 128,
    min_angle_gain_deg: float = 1.0e-8,
    min_normal_dot: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Return a deterministic, nonoverlapping same-parent flip proposal.

    Only candidate pairs touching target_face_rows are considered. The proposal
    changes topology only; callers must run the complete exact geometry,
    fixed-037 area, anchor, source-cap and typed-bank gates.
    """
    _validate_input(candidate)
    if not isinstance(max_flips, int) or max_flips < 0:
        raise FlipInputError("max_flips must be a nonnegative integer")
    if not np.isfinite(min_angle_gain_deg) or min_angle_gain_deg < 0.0:
        raise FlipInputError("minimum angle gain must be finite and nonnegative")
    if not np.isfinite(min_normal_dot) or min_normal_dot < 0.0 or min_normal_dot >= 1.0:
        raise FlipInputError("min_normal_dot must lie in [0,1)")
    f = np.asarray(candidate["source_O_faces"])
    seeds = sorted(set(int(x) for x in target_face_rows))
    if any(row < 0 or row >= len(f) for row in seeds):
        raise FlipInputError("target face row out of range")
    if not seeds:
        raise FlipInputError("at least one target face row is required")
    edges = _edge_map(f)
    evaluated: dict[tuple[int, int], dict] = {}
    reasons: dict[str, int] = {}
    for seed in seeds:
        for a, b in zip(f[seed], np.roll(f[seed], -1)):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            inc = edges[key]
            pair_key = tuple(sorted(x[0] for x in inc)) if len(inc) == 2 else None
            if pair_key is not None and pair_key in evaluated:
                continue
            candidate_pair, reason = _pair_candidate(
                candidate, edges, inc, set(seeds), min_angle_gain_deg, min_normal_dot
            )
            if candidate_pair is not None:
                evaluated[pair_key] = candidate_pair
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
    ranked = sorted(
        evaluated.values(),
        key=lambda op: (
            -op["angle_gain_deg"],
            min(op["seed_face_rows"]),
            op["parent_row"],
            op["face_rows"],
            op["edge"],
        ),
    )
    accepted: list[dict] = []
    used_faces: set[int] = set()
    used_nodes: set[int] = set()
    for op in ranked:
        if len(accepted) >= max_flips:
            break
        faces = set(op["face_rows"])
        nodes = {int(node) for face in op["new_faces"] for node in face}
        if faces & used_faces or nodes & used_nodes:
            reasons["overlaps_selected_batch"] = reasons.get("overlaps_selected_batch", 0) + 1
            continue
        accepted.append(op)
        used_faces |= faces
        used_nodes |= nodes

    out = {key: np.array(value, copy=True) for key, value in candidate.items()}
    old_faces = np.asarray(candidate["source_O_faces"])
    new_faces = np.array(old_faces, copy=True)
    new_ref_faces = np.asarray(candidate["area_reference_037_source_O_faces"]).copy()
    new_bary = np.asarray(candidate["refinement_face_parent_barycentric"]).copy()
    source_ptr = np.asarray(candidate["face_source_ancestry_indptr"])
    source_values = np.asarray(candidate["face_source_ancestry_face_rows"])
    sub_ptr = np.asarray(candidate["face_fine_subface_ancestry_indptr"])
    sub_values = np.asarray(candidate["face_fine_subface_ancestry_rows"])
    source_updates: dict[int, list[int]] = {}
    sub_updates: dict[int, list[int]] = {}
    for op in accepted:
        rows = op["face_rows"]
        bary_by_vertex = {int(k): np.asarray(v, dtype=np.float64)
                          for k, v in op["barycentric_by_vertex"].items()}
        for face_row, tri in zip(rows, op["new_faces"]):
            new_faces[face_row] = np.asarray(tri, dtype=new_faces.dtype)
            new_ref_faces[face_row] = np.asarray(tri, dtype=new_ref_faces.dtype)
            new_bary[face_row] = np.asarray([bary_by_vertex[int(node)] for node in tri])
            source_updates[face_row] = op["source_face_ancestry_union"]
            sub_updates[face_row] = op["fine_subface_ancestry_union"]
        for key in METADATA_KEYS:
            if key in out:
                values = np.asarray(candidate[key])
                left, right = values[rows[0]], values[rows[1]]
                out[key][rows] = left if left == right else -1

    out["source_O_faces"] = new_faces
    out["area_reference_037_source_O_faces"] = new_ref_faces
    out["refinement_face_parent_barycentric"] = new_bary
    if accepted:
        out["face_source_ancestry_indptr"], out["face_source_ancestry_face_rows"] = _rebuild_csr(
            source_ptr, source_values, source_updates, len(old_faces)
        )
        out["face_fine_subface_ancestry_indptr"], out["face_fine_subface_ancestry_rows"] = _rebuild_csr(
            sub_ptr, sub_values, sub_updates, len(old_faces)
        )
    out["same_parent_flip_349_face_rows"] = np.asarray(
        [row for op in accepted for row in op["face_rows"]], dtype=np.int32
    )
    out["same_parent_flip_349_pair_face_rows"] = np.asarray(
        [op["face_rows"] for op in accepted], dtype=np.int32
    ).reshape((-1, 2))
    out["same_parent_flip_349_parent_rows"] = np.asarray(
        [op["parent_row"] for op in accepted], dtype=np.int64
    )
    out["same_parent_flip_349_semantics"] = np.asarray(
        ["fixed-coordinate same-immediate-parent 2-to-2 flip; exact external audits required"]
    )

    before_edges, before_boundary, before_overfull, before_conflicts = _edge_stats(old_faces)
    after_edges, after_boundary, after_overfull, after_conflicts = _edge_stats(new_faces)
    parents = np.asarray(candidate["refinement_face_parent_face_row"])
    before_parent_rings, before_parent_overfull = _parent_rings(old_faces, parents)
    after_parent_rings, after_parent_overfull = _parent_rings(new_faces, parents)
    if before_overfull or after_overfull or before_parent_overfull or after_parent_overfull:
        raise FlipInputError("input/output edge incidence exceeds two globally or within a parent patch")
    if before_boundary != after_boundary:
        raise FlipInputError("flip batch changed the surface boundary edge set")
    if before_parent_rings != after_parent_rings:
        raise FlipInputError("flip batch changed one or more immediate-parent boundary rings")
    if after_conflicts > before_conflicts:
        raise FlipInputError("flip batch increased directed winding conflicts")
    changed_edge_keys = set()
    for op in accepted:
        changed_edge_keys.add(tuple(op["edge"]))
        changed_edge_keys.add(tuple(op["new_diagonal"]))
    for edge in set(before_edges) | set(after_edges):
        if edge in changed_edge_keys:
            continue
        if len(before_edges.get(edge, ())) != len(after_edges.get(edge, ())):
            raise FlipInputError(f"unrelated edge incidence changed at {edge}")
    if not np.array_equal(out["source_O_faces"], out["area_reference_037_source_O_faces"]):
        raise FlipInputError("current and matched-037 face arrays diverged")
    if not np.array_equal(out["points_world_m"], candidate["points_world_m"]):
        raise FlipInputError("point coordinates changed")
    for key, value in candidate.items():
        if key.endswith("anchor_rows") or "anchor_source" in key or "anchor_output" in key:
            if key in out and not np.array_equal(out[key], value):
                raise FlipInputError(f"anchor/source identity array changed: {key}")

    report = {
        "status": "SAME_PARENT_FLIP_PROPOSAL_EMITTED" if accepted else "NO_ELIGIBLE_IMPROVING_FLIPS",
        "geometry_admitted": False,
        "exact_full_surface_self_audit_required": True,
        "fixed_037_area_audit_required": True,
        "all_185_typed_bank_audits_required": True,
        "source_point_rows_unchanged": True,
        "matched_037_face_connectivity_synchronized": True,
        "seed_face_rows": seeds,
        "eligible_improving_pair_count": len(evaluated),
        "accepted_pair_count": len(accepted),
        "rejection_reason_counts": reasons,
        "changed_face_rows": sorted(row for op in accepted for row in op["face_rows"]),
        "boundary_edge_count_before": len(before_boundary),
        "boundary_edge_count_after": len(after_boundary),
        "immediate_parent_boundary_rings_preserved": True,
        "directed_winding_conflicts_before": before_conflicts,
        "directed_winding_conflicts_after": after_conflicts,
        "protected_anchor_rows_preserved": True,
        "operations": accepted,
    }
    return out, report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_target_rows(path: Path) -> list[int]:
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("face_rows")
    if not isinstance(value, list):
        raise FlipInputError("target JSON must be a list or an object with face_rows")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise FlipInputError("target face rows must be exact JSON integers")
    return list(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--target-face-rows-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-flips", type=int, default=128)
    parser.add_argument("--min-angle-gain-deg", type=float, default=1.0e-8)
    args = parser.parse_args()
    actual_sha = sha256_file(args.candidate)
    if actual_sha != args.expected_sha256:
        raise FlipInputError(f"candidate SHA mismatch: {actual_sha}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    with np.load(args.candidate, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]).copy() for key in archive.files}
    result, report = propose_flip_batch(
        source, _read_target_rows(args.target_face_rows_json),
        max_flips=args.max_flips, min_angle_gain_deg=args.min_angle_gain_deg,
    )
    candidate_out = args.output_dir / "candidate.npz"
    with candidate_out.open("xb") as stream:
        np.savez_compressed(stream, **result)
    report.update({
        "input_candidate": str(args.candidate),
        "input_candidate_sha256": actual_sha,
        "target_face_rows_json": str(args.target_face_rows_json),
        "target_face_rows_sha256": sha256_file(args.target_face_rows_json),
        "output_candidate": str(candidate_out),
        "output_candidate_sha256": sha256_file(candidate_out),
        "output_arrays": {
            "points_world_m_sha256": _array_sha(result["points_world_m"]),
            "source_O_faces_sha256": _array_sha(result["source_O_faces"]),
            "area_reference_037_source_O_faces_sha256": _array_sha(result["area_reference_037_source_O_faces"]),
            "refinement_face_parent_face_row_sha256": _array_sha(result["refinement_face_parent_face_row"]),
            "refinement_face_parent_barycentric_sha256": _array_sha(result["refinement_face_parent_barycentric"]),
        },
    })
    (args.output_dir / "flip-proposal-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

