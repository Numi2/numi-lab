#!/usr/bin/env python3
"""Conforming, ancestry-preserving tetrahedral edge bisection for asset prep.

This is a mesh construction primitive, not a collision repair guarantee or a
physical-admission path. A selected edge is bisected once globally; every
incident active tetrahedron is split, with incidence updated as further edges
are processed. Thus a shared face containing a selected edge receives the same
subdivision from both incident cells.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_EDGE_CORNERS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _signed_six_volume(points: Sequence[np.ndarray] | np.ndarray, tet: Sequence[int]) -> float:
    p = np.asarray([points[int(i)] for i in tet], dtype=np.float64)
    return float(np.linalg.det(np.stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]), axis=1)))


def _positive_child(points: Sequence[np.ndarray], tet: Sequence[int]) -> tuple[int, int, int, int]:
    out = list(map(int, tet))
    det = _signed_six_volume(points, out)
    if not np.isfinite(det) or det == 0.0:
        raise ValueError(f"edge split produced zero/nonfinite child volume: tet={out}, det6={det}")
    if det < 0.0:
        out[0], out[1] = out[1], out[0]
        det = -det
    if det <= 0.0:
        raise ValueError(f"edge split produced nonpositive child orientation: tet={out}, det6={det}")
    return tuple(out)


def _validate_source_csr(
    source_indptr: np.ndarray,
    source_indices: np.ndarray,
    source_weights: np.ndarray,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_ptr = np.asarray(source_indptr)
    raw_ids = np.asarray(source_indices)
    if raw_ptr.dtype.kind not in "iu" or raw_ids.dtype.kind not in "iu":
        raise ValueError("source ancestry row pointers and IDs must be integer arrays")
    ptr = raw_ptr.astype(np.int64, copy=False)
    ids = raw_ids.astype(np.int64, copy=False)
    weights = np.asarray(source_weights, dtype=np.float64)
    if (ptr.shape != (node_count + 1,) or ptr[0] != 0 or ptr[-1] != len(ids)
            or ids.ndim != 1 or weights.shape != ids.shape or np.any(np.diff(ptr) < 0)):
        raise ValueError("malformed source-vertex ancestry CSR")
    if np.any(ids < 0) or not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("source ancestry must have nonnegative IDs and finite nonnegative weights")
    for row in range(node_count):
        lo, hi = int(ptr[row]), int(ptr[row + 1])
        if hi > lo and abs(float(np.sum(weights[lo:hi], dtype=np.float64)) - 1.0) > 2e-12:
            raise ValueError(f"source ancestry row {row} does not sum to one")
    return ptr, ids, weights


def _append_midpoint_csr(
    ptr: np.ndarray, ids: np.ndarray, weights: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_ptr = ptr.tolist()
    out_ids = ids.tolist()
    out_weights = weights.tolist()
    for a, b in edges:
        accum: dict[int, float] = {}
        for node in (int(a), int(b)):
            lo, hi = int(ptr[node]), int(ptr[node + 1])
            if lo == hi:
                raise ValueError(f"cannot derive source ancestry for midpoint edge {(int(a), int(b))}: empty endpoint row {node}")
            for source_id, weight in zip(ids[lo:hi], weights[lo:hi]):
                key = int(source_id)
                accum[key] = accum.get(key, 0.0) + 0.5 * float(weight)
        keys = sorted(accum)
        vals = [accum[k] for k in keys]
        total = float(np.sum(vals, dtype=np.float64))
        if not keys or not np.isfinite(vals).all() or min(vals) < 0.0 or abs(total - 1.0) > 2e-12:
            raise ValueError(f"invalid midpoint source ancestry for edge {(int(a), int(b))}: sum={total}")
        out_ids.extend(keys)
        out_weights.extend(vals)
        out_ptr.append(len(out_ids))
    return np.asarray(out_ptr, dtype=np.int64), np.asarray(out_ids, dtype=np.int64), np.asarray(out_weights, dtype=np.float64)


def _volumes_and_quality(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=np.float64)[np.asarray(tets, dtype=np.int64)]
    det6 = np.linalg.det(np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2))
    max_edge = np.zeros(len(p), dtype=np.float64)
    for i, j in _EDGE_CORNERS:
        max_edge = np.maximum(max_edge, np.linalg.norm(p[:, i] - p[:, j], axis=1))
    return det6 / 6.0, det6 / np.maximum(max_edge, np.finfo(np.float64).tiny) ** 3


def refine_tetra_edges(
    reference_points_f32: np.ndarray,
    tetrahedra: np.ndarray,
    marked_edges: Sequence[Sequence[int]],
    *,
    point_fields: Mapping[str, np.ndarray] | None = None,
    material_ids: np.ndarray | None = None,
    parent_cell_ids: np.ndarray | None = None,
    source_vertex_indptr: np.ndarray | None = None,
    source_vertex_indices: np.ndarray | None = None,
    source_vertex_weights: np.ndarray | None = None,
    minimum_normalized_quality: float | None = None,
    maximum_parent_reference_volume_relative_drift: float | None = 1e-4,
) -> dict[str, Any]:
    """Bisect selected edges globally and return ordinary mesh arrays.

    Args:
        reference_points_f32: The source/reference point frame. New midpoint
            positions and parent-volume fractions are defined in this frame.
        tetrahedra: Mx4 input cell rows. Every input row must have positive
            signed volume in ``reference_points_f32``.
        marked_edges: Input node-ID pairs. They are sorted/deduplicated before
            processing; each must occur in at least one input tetrahedron.
        point_fields: Optional named Nx3 coordinate arrays in other frames
            (for example accepted/current or donor-local). New points are the
            Float32/Float64 arithmetic endpoint midpoints as appropriate; the
            old N rows are preserved exactly.
        material_ids: Optional per-input-cell material labels. Children inherit
            the label of their input parent cell.
        parent_cell_ids: Optional stable per-input-cell IDs. The output keeps
            both ``child_parent_rows`` (dense input row) and
            ``child_parent_cell_ids`` (stable caller IDs).
        source_vertex_*: Optional CSR source ancestry. New midpoint rows are the
            normalized 0.5/0.5 combination of the two endpoint rows.
        minimum_normalized_quality: Optional hard postcondition. Quality is
            signed six-volume divided by maximum-edge-length cubed.
        maximum_parent_reference_volume_relative_drift: Hard per-parent
            reference-volume closure limit after Float32 midpoint emission.
            The default is 1e-4 relative. ``None`` disables the rejection
            only for an explicitly diagnostic run; raw child/input-parent
            ratios remain available and must not be interpreted as exact
            overlap weights.

    Returns a dict of NumPy arrays. ``child_reference_volume_fraction_of_input_parent``
    is the raw emitted-child volume divided by its input parent's reference
    volume. ``parent_reference_volume_fraction`` is normalized within each
    input parent for conservative bookkeeping, not exact geometric-overlap
    transfer. By default the function rejects parent volume drift above
    ``maximum_parent_reference_volume_relative_drift`` before returning either
    fraction array.
    """
    X = np.asarray(reference_points_f32)
    raw_tets = np.asarray(tetrahedra)
    if raw_tets.dtype.kind not in "iu":
        raise ValueError("tetrahedra must use integer node IDs")
    T = raw_tets.astype(np.int64, copy=False)
    if X.dtype != np.float32 or X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all():
        raise ValueError("reference_points_f32 must be finite Nx3 Float32")
    if T.ndim != 2 or T.shape[1] != 4 or len(T) == 0:
        raise ValueError("tetrahedra must be a nonempty Mx4 table")
    if np.any(T < 0) or np.any(T >= len(X)) or any(len(set(map(int, row))) != 4 for row in T):
        raise ValueError("tetrahedra contain invalid or repeated node IDs")
    old_volumes, old_quality = _volumes_and_quality(X, T)
    if not np.isfinite(old_volumes).all() or np.any(old_volumes <= 0.0):
        raise ValueError("all input tetrahedra must have positive finite signed reference volume")
    if not np.isfinite(old_quality).all() or np.any(old_quality <= 0.0):
        raise ValueError("all input tetrahedra must have positive finite normalized quality")

    M = len(T)
    if parent_cell_ids is None:
        cell_ids = np.arange(M, dtype=np.int64)
    else:
        raw_cell_ids = np.asarray(parent_cell_ids)
        if raw_cell_ids.dtype.kind not in "iu":
            raise ValueError("parent_cell_ids must use integer source IDs")
        cell_ids = raw_cell_ids.astype(np.int64, copy=False)
    if cell_ids.shape != (M,):
        raise ValueError("parent_cell_ids must have one value per input tetrahedron")
    mats = None if material_ids is None else np.asarray(material_ids)
    if mats is not None and mats.shape != (M,):
        raise ValueError("material_ids must have one value per input tetrahedron")

    fields: dict[str, np.ndarray] = {}
    for name, value in (point_fields or {}).items():
        if not isinstance(name, str) or not name or name in fields:
            raise ValueError("point field names must be unique nonempty strings")
        array = np.asarray(value)
        if array.shape != X.shape or array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ValueError(f"point field {name!r} must be finite Nx3 numeric coordinates")
        fields[name] = array.copy()

    ancestry_args = (source_vertex_indptr, source_vertex_indices, source_vertex_weights)
    if all(x is None for x in ancestry_args):
        source_csr = None
    elif any(x is None for x in ancestry_args):
        raise ValueError("source ancestry requires indptr, indices, and weights together")
    else:
        source_csr = _validate_source_csr(*ancestry_args, len(X))

    normalized_edges: list[tuple[int, int]] = []
    for edge in marked_edges:
        raw_edge = np.asarray(edge)
        if raw_edge.shape != (2,) or raw_edge.dtype.kind not in "iu":
            raise ValueError(f"marked edge must have two node IDs: {edge!r}")
        a, b = sorted((int(raw_edge[0]), int(raw_edge[1])))
        if a < 0 or b >= len(X) or a == b:
            raise ValueError(f"invalid marked edge {(a, b)}")
        normalized_edges.append((a, b))
    marks = sorted(set(normalized_edges))
    if minimum_normalized_quality is not None and (not np.isfinite(minimum_normalized_quality) or minimum_normalized_quality <= 0.0):
        raise ValueError("minimum_normalized_quality must be finite and positive")
    if maximum_parent_reference_volume_relative_drift is not None and (
        not np.isfinite(maximum_parent_reference_volume_relative_drift)
        or maximum_parent_reference_volume_relative_drift < 0.0
    ):
        raise ValueError("maximum_parent_reference_volume_relative_drift must be finite and nonnegative, or None")

    records = [tuple(map(int, row)) for row in T]
    active = bytearray([1]) * M
    parent_rows = list(range(M))
    points = list(X)
    incidence: dict[tuple[int, int], set[int]] = {edge: set() for edge in marks}
    for row, tet in enumerate(records):
        for i, j in _EDGE_CORNERS:
            edge = tuple(sorted((tet[i], tet[j])))
            if edge in incidence:
                incidence[edge].add(row)
    absent = [edge for edge in marks if not incidence[edge]]
    if absent:
        raise ValueError(f"marked edges are absent from tetrahedra: {absent[:8]}")

    midpoint_edges: list[tuple[int, int]] = []
    midpoint_rows: list[int] = []
    touched_parent_rows: set[int] = set()
    midpoint_rounding_m = []
    midpoint_off_edge_m = []
    for edge in marks:
        incident = sorted(incidence[edge])
        if not incident:
            raise ValueError(f"marked edge lost before split: {edge}")
        a, b = edge
        midpoint = np.asarray((X[a].astype(np.float64) + X[b].astype(np.float64)) * 0.5, dtype=np.float32)
        if np.array_equal(midpoint, X[a]) or np.array_equal(midpoint, X[b]):
            raise ValueError(f"Float32 midpoint collapsed to endpoint: edge={edge}")
        midrow = len(points)
        points.append(midpoint)
        midpoint_rows.append(midrow)
        midpoint_edges.append(edge)
        ideal = (X[a].astype(np.float64) + X[b].astype(np.float64)) * 0.5
        delta = midpoint.astype(np.float64) - ideal
        direction = X[b].astype(np.float64) - X[a].astype(np.float64)
        direction_sq = float(np.dot(direction, direction))
        projection = direction * (float(np.dot(delta, direction)) / direction_sq) if direction_sq > 0.0 else np.zeros(3)
        midpoint_rounding_m.append(float(np.linalg.norm(delta)))
        midpoint_off_edge_m.append(float(np.linalg.norm(delta - projection)))
        for oldrow in incident:
            if not active[oldrow]:
                raise ValueError(f"inactive tetrahedron in edge incidence: edge={edge} row={oldrow}")
            old = records[oldrow]
            others = [v for v in old if v not in edge]
            if len(others) != 2:
                raise ValueError(f"malformed edge/tetrahedron incidence: edge={edge} row={oldrow}")
            touched_parent_rows.add(parent_rows[oldrow])
            for i, j in _EDGE_CORNERS:
                pending_edge = tuple(sorted((old[i], old[j])))
                if pending_edge in incidence:
                    incidence[pending_edge].discard(oldrow)
            active[oldrow] = 0
            c, d = others
            children = (
                _positive_child(points, (a, midrow, c, d)),
                _positive_child(points, (midrow, b, c, d)),
            )
            for child in children:
                newrow = len(records)
                records.append(child)
                parent_rows.append(parent_rows[oldrow])
                active.append(1)
                for i, j in _EDGE_CORNERS:
                    child_edge = tuple(sorted((child[i], child[j])))
                    if child_edge in incidence:
                        incidence[child_edge].add(newrow)
        if incidence[edge]:
            raise ValueError(f"edge not fully removed after conforming split: {edge}")

    live_rows = [row for row, is_active in enumerate(active) if is_active]
    out_tets = np.asarray([records[row] for row in live_rows], dtype=np.int64)
    out_parent = np.asarray([parent_rows[row] for row in live_rows], dtype=np.int64)
    out_points = np.asarray(points, dtype=np.float32)
    out_volumes, out_quality = _volumes_and_quality(out_points, out_tets)
    if not np.isfinite(out_volumes).all() or np.any(out_volumes <= 0.0):
        raise ValueError("refined mesh contains nonpositive or nonfinite reference-volume cells")
    if not np.isfinite(out_quality).all() or np.any(out_quality <= 0.0):
        raise ValueError("refined mesh contains nonpositive or nonfinite quality")
    if minimum_normalized_quality is not None and float(out_quality.min()) < float(minimum_normalized_quality):
        raise ValueError(
            f"refined minimum quality {float(out_quality.min()):.9g} is below required "
            f"{float(minimum_normalized_quality):.9g}"
        )

    parent_volume_sum = np.bincount(out_parent, weights=out_volumes, minlength=M)
    if np.any(parent_volume_sum <= 0.0):
        raise ValueError("one or more input cells have no refined children")
    relative_drift = np.abs(parent_volume_sum - old_volumes) / old_volumes
    if maximum_parent_reference_volume_relative_drift is not None:
        worst_row = int(np.argmax(relative_drift))
        worst_drift = float(relative_drift[worst_row])
        if worst_drift > float(maximum_parent_reference_volume_relative_drift):
            raise ValueError(
                "per-parent reference-volume closure failed: "
                f"row={worst_row} relative_drift={worst_drift:.17g} exceeds "
                f"limit={float(maximum_parent_reference_volume_relative_drift):.17g}"
            )
    raw_parent_fractions = out_volumes / old_volumes[out_parent]
    fractions = out_volumes / parent_volume_sum[out_parent]
    fraction_sum = np.bincount(out_parent, weights=fractions, minlength=M)

    out_fields: dict[str, np.ndarray] = {}
    edge_array = np.asarray(midpoint_edges, dtype=np.int64).reshape((-1, 2))
    for name, array in fields.items():
        mids = ((array[edge_array[:, 0]].astype(np.float64) + array[edge_array[:, 1]].astype(np.float64)) * 0.5).astype(array.dtype) if len(edge_array) else np.empty((0, 3), dtype=array.dtype)
        out_fields[name] = np.concatenate((array, mids), axis=0)

    result: dict[str, Any] = {
        "reference_points_f32": out_points,
        "point_fields": out_fields,
        "tetrahedra": out_tets,
        "child_parent_rows": out_parent,
        "child_parent_cell_ids": cell_ids[out_parent],
        "material_ids": None if mats is None else mats[out_parent],
        "child_reference_volume_m3": out_volumes,
        "parent_reference_volume_before_m3": old_volumes,
        "parent_reference_volume_after_m3": parent_volume_sum,
        "parent_reference_volume_relative_drift": relative_drift,
        "child_reference_volume_fraction_of_input_parent": raw_parent_fractions,
        "parent_reference_volume_fraction": fractions,
        "new_node_parent_edges": edge_array,
        "new_node_rows": np.asarray(midpoint_rows, dtype=np.int64),
    }
    if source_csr is not None:
        p2, i2, w2 = _append_midpoint_csr(*source_csr, edge_array)
        result["source_vertex_ancestry_indptr"] = p2
        result["source_vertex_ancestry_indices"] = i2
        result["source_vertex_ancestry_weights"] = w2
    result["report"] = {
        "input_node_count": int(len(X)),
        "output_node_count": int(len(out_points)),
        "input_tetrahedron_count": int(M),
        "output_tetrahedron_count": int(len(out_tets)),
        "selected_unique_edge_count": int(len(marks)),
        "touched_input_tetrahedron_count": int(len(touched_parent_rows)),
        "minimum_reference_normalized_quality": float(out_quality.min()),
        "minimum_input_reference_normalized_quality": float(old_quality.min()),
        "minimum_parent_fraction_sum": float(fraction_sum.min()),
        "maximum_parent_fraction_sum_error": float(np.max(np.abs(fraction_sum - 1.0))),
        "maximum_parent_reference_volume_relative_drift": float(relative_drift.max()),
        "parent_reference_volume_relative_drift_limit": (
            None if maximum_parent_reference_volume_relative_drift is None
            else float(maximum_parent_reference_volume_relative_drift)
        ),
        "parent_reference_volume_closure_pass": (
            True if maximum_parent_reference_volume_relative_drift is None
            else bool(float(relative_drift.max()) <= float(maximum_parent_reference_volume_relative_drift))
        ),
        "total_reference_volume_relative_drift": float(abs(float(out_volumes.sum()) - float(old_volumes.sum())) / float(old_volumes.sum())),
        "maximum_midpoint_rounding_displacement_m": float(max(midpoint_rounding_m, default=0.0)),
        "maximum_midpoint_off_edge_distance_m": float(max(midpoint_off_edge_m, default=0.0)),
        "source_ancestry_appended_rows": int(len(edge_array)) if source_csr is not None else 0,
        "material_ids_propagated": bool(mats is not None),
        "parent_cell_ids_propagated": True,
        "all_output_reference_volumes_positive": True,
        "normalized_quality_floor": None if minimum_normalized_quality is None else float(minimum_normalized_quality),
        "normalized_quality_floor_pass": True if minimum_normalized_quality is None else bool(float(out_quality.min()) >= float(minimum_normalized_quality)),
        "refinement_is_physical_admission": False,
    }
    return result


def refine_tetra_edges_f64_reference(canonical_reference_points_f64, current_points_f32, tetrahedra, marked_edges, *, point_fields=None, material_ids=None, parent_cell_ids=None, source_vertex_indptr=None, source_vertex_indices=None, source_vertex_weights=None, minimum_reference_f32_normalized_quality=1e-5, minimum_current_f32_normalized_quality=1e-5, maximum_f64_parent_reference_volume_relative_drift=1e-4):
    """Bisect in canonical Float64 reference and gate emitted Float32 quality separately."""
    ref = np.asarray(canonical_reference_points_f64)
    current = np.asarray(current_points_f32)
    if ref.dtype != np.float64 or ref.ndim != 2 or ref.shape[1] != 3 or not np.isfinite(ref).all():
        raise ValueError("canonical_reference_points_f64 must be finite Nx3 Float64")
    if current.dtype != np.float32 or current.shape != ref.shape or not np.isfinite(current).all():
        raise ValueError("current_points_f32 must be finite Nx3 Float32 with the reference shape")
    if not np.isfinite(maximum_f64_parent_reference_volume_relative_drift) or maximum_f64_parent_reference_volume_relative_drift < 0:
        raise ValueError("maximum_f64_parent_reference_volume_relative_drift must be finite and nonnegative")
    for name, floor in (("reference F32 quality floor", minimum_reference_f32_normalized_quality), ("current F32 quality floor", minimum_current_f32_normalized_quality)):
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError(name + " must be finite and positive")
    old_ref_v, old_ref_q = _volumes_and_quality(ref, tetrahedra)
    old_cur_v, old_cur_q = _volumes_and_quality(current, tetrahedra)
    if not np.isfinite(old_ref_v).all() or np.any(old_ref_v <= 0) or not np.isfinite(old_ref_q).all() or np.any(old_ref_q <= 0):
        raise ValueError("canonical F64 reference input must have positive finite volume and quality")
    if not np.isfinite(old_cur_v).all() or np.any(old_cur_v <= 0) or not np.isfinite(old_cur_q).all() or np.any(old_cur_q <= 0):
        raise ValueError("current F32 input must have positive finite volume and quality")
    fields = dict(point_fields or {})
    if "canonical_reference_points_f64" in fields or "current_points_f32" in fields:
        raise ValueError("point_fields cannot replace canonical reference or current coordinates")
    fields["canonical_reference_points_f64"] = ref
    fields["current_points_f32"] = current
    result = refine_tetra_edges(ref.astype(np.float32), tetrahedra, marked_edges, point_fields=fields,
        material_ids=material_ids, parent_cell_ids=parent_cell_ids,
        source_vertex_indptr=source_vertex_indptr, source_vertex_indices=source_vertex_indices,
        source_vertex_weights=source_vertex_weights, minimum_normalized_quality=None,
        maximum_parent_reference_volume_relative_drift=None)
    ref64_out = result["point_fields"]["canonical_reference_points_f64"]
    cur_out = result["point_fields"]["current_points_f32"]
    ref32_out = result["reference_points_f32"]
    child_tets, parent_rows = result["tetrahedra"], result["child_parent_rows"]
    ref_v, ref_q = _volumes_and_quality(ref64_out, child_tets)
    ref32_v, ref32_q = _volumes_and_quality(ref32_out, child_tets)
    cur_v, cur_q = _volumes_and_quality(cur_out, child_tets)
    if any(not np.isfinite(x).all() for x in (ref_v, ref_q, ref32_v, ref32_q, cur_v, cur_q)):
        raise ValueError("refined reference/current volume or quality is nonfinite")
    if np.any(ref_v <= 0) or np.any(ref32_v <= 0) or np.any(cur_v <= 0):
        raise ValueError("refinement emitted a nonpositive canonical F64 or F32 cell")
    parent_after = np.bincount(parent_rows, weights=ref_v, minlength=len(old_ref_v))
    if np.any(parent_after <= 0):
        raise ValueError("an input parent has no refined children")
    drift = np.abs(parent_after - old_ref_v) / old_ref_v
    total_drift = abs(float(ref_v.sum()) - float(old_ref_v.sum())) / float(old_ref_v.sum())
    if float(drift.max()) > maximum_f64_parent_reference_volume_relative_drift:
        row = int(np.argmax(drift))
        raise ValueError(f"canonical F64 parent-volume closure failed at row {row}: {float(drift[row]):.17g}")
    if float(ref32_q.min()) < minimum_reference_f32_normalized_quality:
        raise ValueError(f"emitted F32 reference quality failed: {float(ref32_q.min()):.17g}")
    if float(cur_q.min()) < minimum_current_f32_normalized_quality:
        raise ValueError(f"emitted F32 current quality failed: {float(cur_q.min()):.17g}")
    old_ref32_v, _ = _volumes_and_quality(ref.astype(np.float32), tetrahedra)
    parent32_after = np.bincount(parent_rows, weights=ref32_v, minlength=len(old_ref32_v))
    f32_drift = np.abs(parent32_after - old_ref32_v) / old_ref32_v
    f32_fraction = ref32_v / old_ref32_v[parent_rows]
    f32_fraction_sum = np.bincount(parent_rows, weights=f32_fraction, minlength=len(old_ref32_v))
    raw_fraction = ref_v / old_ref_v[parent_rows]
    fraction_sum = np.bincount(parent_rows, weights=raw_fraction, minlength=len(old_ref_v))
    result.update({
        "canonical_reference_points_f64": ref64_out, "current_points_f32": cur_out,
        "child_canonical_reference_volume_m3_f64": ref_v,
        "child_emitted_reference_volume_m3_f32": ref32_v,
        "parent_emitted_reference_volume_before_m3_f32": old_ref32_v,
        "parent_emitted_reference_volume_after_m3_f32": parent32_after,
        "parent_emitted_reference_volume_relative_drift_f32": f32_drift,
        "child_emitted_reference_volume_fraction_of_input_parent_f32": f32_fraction,
        "parent_canonical_reference_volume_before_m3_f64": old_ref_v,
        "parent_canonical_reference_volume_after_m3_f64": parent_after,
        "parent_canonical_reference_volume_relative_drift_f64": drift,
        "child_canonical_reference_volume_fraction_of_input_parent_f64": raw_fraction,
        "reference_f32_normalized_quality": ref32_q, "current_f32_normalized_quality": cur_q,
        "canonical_reference_normalized_quality_f64": ref_q,
        "report": {**result["report"],
            "reference_frame_contract": "canonical owner Float64; midpoint constructed in Float64 before Float32 emission",
            "canonical_f64_parent_reference_volume_closure_pass": True,
            "canonical_f64_parent_reference_volume_relative_drift_limit": float(maximum_f64_parent_reference_volume_relative_drift),
            "maximum_canonical_f64_parent_reference_volume_relative_drift": float(drift.max()),
            "maximum_total_canonical_f64_reference_volume_relative_drift": float(total_drift),
            "maximum_raw_canonical_f64_parent_fraction_sum_error": float(np.max(np.abs(fraction_sum - 1.0))),
            "minimum_emitted_reference_f32_normalized_quality": float(ref32_q.min()),
            "minimum_emitted_reference_f32_normalized_quality_required": float(minimum_reference_f32_normalized_quality),
            "minimum_emitted_current_f32_normalized_quality": float(cur_q.min()),
            "minimum_emitted_current_f32_normalized_quality_required": float(minimum_current_f32_normalized_quality),
            "emitted_f32_reference_quality_pass": True, "emitted_f32_current_quality_pass": True,
            "legacy_f32_reference_volume_drift": float(result["report"]["maximum_parent_reference_volume_relative_drift"]),
            "emitted_f32_reference_parent_volume_max_relative_drift": float(f32_drift.max()),
            "emitted_f32_reference_parent_fraction_sum_error": float(np.max(np.abs(f32_fraction_sum - 1.0))),
            "emitted_f32_parent_volume_closure_is_diagnostic_only": True,
            "legacy_f32_reference_volume_drift_is_diagnostic_only": True,
            "canonical_f64_volume_fractions_are_overlap_transfer": False, "physical_mass_transfer_admission": False}})
    return result
