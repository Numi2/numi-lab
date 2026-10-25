from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
from typing import Any

import numpy as np

class CertificationError(ValueError):
    """Invalid inputs or incomplete source ancestry; no contact plane may be emitted."""


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape or not np.issubdtype(arr.dtype, np.number):
        raise CertificationError(f"{name} must have shape {shape}")
    arr = arr.astype(np.float64, copy=False)
    if not np.isfinite(arr).all():
        raise CertificationError(f"{name} contains nonfinite values")
    return arr


def _unit(v: np.ndarray, name: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= np.finfo(np.float64).tiny:
        raise CertificationError(f"{name} has zero or invalid length")
    return v / n


def build_source_outward_field(
    *,
    source_global_node_ids: np.ndarray,
    output_source_input_row: np.ndarray,
    outer_points_by_column_m: np.ndarray,
    deep_points_by_column_m: np.ndarray,
    native_parent_tetrahedra: np.ndarray,
    generated_output_rows: np.ndarray,
    generated_parent_tet_rows: np.ndarray,
    generated_parent_barycentric: np.ndarray,
    column_count: int = 2411,
    direction_tolerance: float = 1e-8,
    strict_parent_agreement: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the declared O-D source field and barycentrically prolong it.

    Direct output rows use their exact source-global node ID. Generated rows
    require every declared native-parent choice and must agree in direction.
    This is a construction field only, not an anatomical outward certificate.
    """
    source_ids = np.asarray(source_global_node_ids)
    input_rows = np.asarray(output_source_input_row)
    parent_tets = np.asarray(native_parent_tetrahedra)
    gen_rows = np.asarray(generated_output_rows)
    gen_parent_rows = np.asarray(generated_parent_tet_rows)
    if not np.issubdtype(source_ids.dtype, np.integer) or source_ids.ndim != 1 or len(source_ids) == 0:
        raise CertificationError("source_global_node_ids must be a nonempty integer vector")
    if not np.issubdtype(input_rows.dtype, np.integer) or input_rows.ndim != 1:
        raise CertificationError("output_source_input_row must be an integer vector")
    if not np.issubdtype(parent_tets.dtype, np.integer) or parent_tets.ndim != 2 or parent_tets.shape[1] != 4:
        raise CertificationError("native_parent_tetrahedra must be integer Mx4")
    if not np.issubdtype(gen_rows.dtype, np.integer) or gen_rows.ndim != 1:
        raise CertificationError("generated_output_rows must be an integer vector")
    if not np.issubdtype(gen_parent_rows.dtype, np.integer) or gen_parent_rows.ndim != 1:
        raise CertificationError("generated_parent_tet_rows must be an integer vector")
    bary = np.asarray(generated_parent_barycentric)
    if bary.shape != (len(gen_rows), 4) or len(gen_parent_rows) != len(gen_rows):
        raise CertificationError("generated ancestry arrays have inconsistent shape")
    bary = bary.astype(np.float64, copy=False)
    if not np.isfinite(bary).all():
        raise CertificationError("generated ancestry contains nonfinite barycentrics")
    outer = np.asarray(outer_points_by_column_m, dtype=np.float64)
    deep = np.asarray(deep_points_by_column_m, dtype=np.float64)
    if outer.shape != (column_count, 3) or deep.shape != outer.shape:
        raise CertificationError("outer/deep source points must be column_count x 3")
    if not np.isfinite(outer).all() or not np.isfinite(deep).all():
        raise CertificationError("outer/deep source points contain nonfinite values")
    if len(np.unique(source_ids)) != len(source_ids):
        raise CertificationError("source_global_node_ids are not unique")
    if np.any(source_ids < 0):
        raise CertificationError("source global node IDs must be nonnegative identity labels")
    if np.any(input_rows < -1) or np.any(input_rows >= len(source_ids)):
        raise CertificationError("output source-input row is outside the declared source domain")
    if np.any(parent_tets < 0) or np.any(parent_tets >= len(source_ids)):
        raise CertificationError("native parent tetrahedron references an unknown source node")
    if np.any(gen_rows < 0) or np.any(gen_rows >= len(input_rows)):
        raise CertificationError("generated output row is out of range")
    if np.any(input_rows[gen_rows] != -1):
        raise CertificationError("generated ancestry points at a direct source row")
    if np.any(gen_parent_rows < 0) or np.any(gen_parent_rows >= len(parent_tets)):
        raise CertificationError("generated ancestry references an unknown native parent tet")
    sums = bary.sum(axis=1)
    bary_tolerance = 1e-10
    if np.any(bary < -bary_tolerance) or np.any(bary > 1.0 + bary_tolerance) or np.any(np.abs(sums - 1.0) > bary_tolerance):
        raise CertificationError("generated parent barycentrics exceed the declared ancestry roundoff tolerance")
    bary_min = float(np.min(bary)) if bary.size else 0.0
    bary_max_sum_error = float(np.max(np.abs(sums - 1.0))) if sums.size else 0.0

    # Stable/global IDs identify source vertices; topology and the O/I/D column
    # convention are indexed by source-input row.
    source_vectors = np.empty((len(source_ids), 3), dtype=np.float64)
    for i, global_id in enumerate(source_ids.astype(np.int64)):
        source_vectors[i] = _unit(outer[i % column_count] - deep[i % column_count],
                                  f"source O-D direction at input row {i} (global ID {int(global_id)})")

    field = np.full((len(input_rows), 3), np.nan, dtype=np.float64)
    direct = np.flatnonzero(input_rows >= 0)
    if len(direct) != len(source_ids):
        raise CertificationError("direct output source rows do not cover the full declared source domain")
    mapped_input = input_rows[direct].astype(np.int64)
    if len(np.unique(mapped_input)) != len(source_ids) or set(map(int, mapped_input)) != set(range(len(source_ids))):
        raise CertificationError("direct output source rows are incomplete or duplicated")
    field[direct] = source_vectors[mapped_input]

    by_generated: dict[int, list[tuple[int, np.ndarray]]] = {}
    for out_row, parent_row, weights in zip(gen_rows.astype(np.int64), gen_parent_rows.astype(np.int64), bary):
        parent_input_rows = parent_tets[int(parent_row)].astype(np.int64)
        parent_vectors = source_vectors[parent_input_rows]
        candidate = weights @ parent_vectors
        candidate = _unit(candidate, f"prolonged source direction at output row {int(out_row)}")
        by_generated.setdefault(int(out_row), []).append((int(parent_row), candidate))
    missing = [int(r) for r in np.flatnonzero(input_rows == -1) if int(r) not in by_generated]
    if missing:
        raise CertificationError(f"generated rows lack parent ancestry: {missing[:8]}")
    direction_spread = 0.0
    direction_disagreements = []
    for row, parent_candidates in by_generated.items():
        reference = parent_candidates[0][1]
        spread = max((float(np.linalg.norm(candidate - reference)) for _, candidate in parent_candidates), default=0.0)
        direction_spread = max(direction_spread, spread)
        if spread > direction_tolerance:
            direction_disagreements.append({
                "output_row": int(row),
                "parent_tet_rows": [int(parent_row) for parent_row, _ in parent_candidates],
                "unit_parent_directions": [candidate.tolist() for _, candidate in parent_candidates],
                "maximum_direction_difference": float(spread),
                "tolerance": float(direction_tolerance),
            })
            if strict_parent_agreement:
                raise CertificationError(
                    f"generated row {row} has disagreeing declared parent directions "
                    f"(spread={spread:.9g}, tolerance={direction_tolerance:.9g})"
                )
            continue
        field[row] = reference
    unresolved_rows = sorted({int(r) for r in missing} | {int(r["output_row"]) for r in direction_disagreements})
    if any(not np.isfinite(field[i]).all() for i in range(len(field)) if i not in unresolved_rows):
        raise CertificationError("source direction field has uninitialized non-conflicting output rows")
    return field.astype(np.float64), {
        "source_direction_definition": "normalize(O_world_m[col]-D_world_m[col]), col=source_global_node_id%column_count",
        "generated_node_extension": "barycentric interpolation over every declared original native parent tetrahedron",
        "source_rows": int(len(source_ids)),
        "direct_output_rows": int(len(direct)),
        "generated_output_rows": int(len(missing) + len(by_generated)),
        "generated_ancestry_choices": int(len(gen_rows)),
        "generated_parent_barycentric_min": bary_min,
        "generated_parent_barycentric_max_row_sum_error": bary_max_sum_error,
        "generated_parent_barycentric_roundoff_tolerance": bary_tolerance,
        "source_global_ids_are_identity_labels": True,
        "topology_indices_use_source_input_rows": True,
        "generated_direction_disagreements": direction_disagreements,
        "unresolved_output_rows": unresolved_rows,
        "strict_parent_agreement": bool(strict_parent_agreement),
        "maximum_generated_parent_direction_spread": float(direction_spread),
        "direction_agreement_tolerance": float(direction_tolerance),
        "anatomical_outward_claim": False,
    }


def interpolate_source_outward_direction(
    node_field: np.ndarray, node_a: int, node_b: int, t: float
) -> np.ndarray:
    field = np.asarray(node_field, dtype=np.float64)
    if field.ndim != 2 or field.shape[1] != 3 or not np.isfinite(field).all():
        raise CertificationError("node field must be finite Nx3")
    if not 0 <= int(node_a) < len(field) or not 0 <= int(node_b) < len(field):
        raise CertificationError("sample endpoint node is outside the mechanics domain")
    if not math.isfinite(float(t)) or not 0.0 <= float(t) <= 1.0:
        raise CertificationError("sample interpolation parameter must lie in [0,1]")
    return _unit((1.0 - float(t)) * field[int(node_a)] + float(t) * field[int(node_b)],
                 "interpolated O-D sample direction")


def ray_endpoint_outside_aabb(
    start_local: np.ndarray,
    direction_local: np.ndarray,
    bank_aabb_min_local: np.ndarray,
    bank_aabb_max_local: np.ndarray,
    *,
    margin_m: float = 2e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extend an inside ray just beyond the first bank-AABB exit."""
    start = _finite_array(start_local, (3,), "ray start")
    direction = _unit(_finite_array(direction_local, (3,), "local ray direction"), "local ray direction")
    lo = _finite_array(bank_aabb_min_local, (3,), "bank AABB minimum")
    hi = _finite_array(bank_aabb_max_local, (3,), "bank AABB maximum")
    if np.any(hi <= lo):
        raise CertificationError("bank AABB must have positive extent on all axes")
    if not math.isfinite(float(margin_m)) or margin_m <= 0:
        raise CertificationError("ray AABB margin must be finite and positive")
    if np.any(start < lo) or np.any(start > hi):
        raise CertificationError("inside query start lies outside the target bank AABB")
    crossings = []
    for axis, d in enumerate(direction):
        if d > 0.0:
            crossings.append(((hi[axis] - start[axis]) / d, axis))
        elif d < 0.0:
            crossings.append(((lo[axis] - start[axis]) / d, axis))
    if not crossings:
        raise CertificationError("ray direction has no AABB crossing axis")
    t_boundary, axis = min(crossings, key=lambda item: item[0])
    if not math.isfinite(float(t_boundary)) or t_boundary < 0.0:
        raise CertificationError("invalid ray/AABB crossing parameter")
    scale = max(float(np.max(np.abs(lo))), float(np.max(np.abs(hi))), float(np.max(np.abs(start))), 1e-3)
    padding = max(float(margin_m), 64.0 * np.finfo(np.float32).eps * scale)
    for _ in range(12):
        end = start + (float(t_boundary) + padding) * direction
        if np.any(end < lo) or np.any(end > hi):
            return end, {
                "aabb_exit_axis": int(axis),
                "aabb_boundary_parameter_m": float(t_boundary),
                "outside_padding_m": float(padding),
                "endpoint_outside_aabb": True,
                "endpoint_local": end.tolist(),
            }
        padding *= 2.0
    raise CertificationError("could not represent a strict outside-AABB ray endpoint")


def _exact_t_barycentric(start: np.ndarray, end: np.ndarray, triangle: np.ndarray) -> tuple[Fraction, tuple[Fraction, Fraction, Fraction]]:
    def fvec(v):
        return tuple(Fraction.from_float(float(x)) for x in v)
    def sub(a, b):
        return tuple(x-y for x, y in zip(a, b))
    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    def dot(a, b):
        return sum((x*y for x, y in zip(a, b)), Fraction(0))
    a, b = fvec(start), fvec(end)
    v0, v1, v2 = map(fvec, triangle)
    d = sub(b, a)
    e1, e2 = sub(v1, v0), sub(v2, v0)
    normal = cross(e1, e2)
    denominator = dot(normal, d)
    if denominator == 0:
        raise CertificationError("exact proper hit unexpectedly has parallel segment")
    t = dot(normal, sub(v0, a)) / denominator
    if not 0 < t < 1:
        raise CertificationError("exact hit parameter is not strictly interior")
    hit = tuple(a[i] + t*d[i] for i in range(3))
    axis = max(range(3), key=lambda i: abs(normal[i]))
    uv = [i for i in range(3) if i != axis]
    p = sub(hit, v0)
    u, v = uv
    area = e1[u]*e2[v] - e1[v]*e2[u]
    if area == 0:
        raise CertificationError("exact proper hit unexpectedly has degenerate projected area")
    b1 = (p[u]*e2[v] - p[v]*e2[u]) / area
    b2 = (e1[u]*p[v] - e1[v]*p[u]) / area
    return t, (1-b1-b2, b1, b2)


def _exact_dot_sign(a: np.ndarray, b: np.ndarray) -> int:
    fa = [Fraction.from_float(float(x)) for x in a]
    fb = [Fraction.from_float(float(x)) for x in b]
    value = sum((x*y for x, y in zip(fa, fb)), Fraction(0))
    return 1 if value > 0 else (-1 if value < 0 else 0)


def certify_first_outward_exit(
    *,
    start_status: str,
    start_local: np.ndarray,
    end_local: np.ndarray,
    source_direction_local: np.ndarray,
    triangles_local: np.ndarray,
    triangle_stable_ids: np.ndarray,
    bank_aabb_min_local: np.ndarray,
    bank_aabb_max_local: np.ndarray,
    expected_bank_triangle_count: int,
    world_start: np.ndarray,
    world_to_local: np.ndarray,
    clearance_m: float = 5e-5,
    exact_classifier=None,
    exact_classifier_sha256: str | None = None,
) -> dict[str, Any]:
    """Select the first exact proper exit along a source-prolonged ray.

    The endpoint must be provably outside the complete bank AABB. Every
    candidate exact hit is retained. Boundary, coplanar, and degenerate
    witnesses reject the result rather than being guessed or skipped.
    """
    start = _finite_array(start_local, (3,), "ray start")
    end = _finite_array(end_local, (3,), "ray end")
    dlocal = _unit(_finite_array(source_direction_local, (3,), "source direction local"),
                   "source direction local")
    pworld = _finite_array(world_start, (3,), "ray world start")
    M = _finite_array(world_to_local, (3, 3), "world-to-local matrix")
    if not math.isfinite(float(np.linalg.cond(M))) or np.linalg.cond(M) > 1e12:
        raise CertificationError("world-to-local matrix is singular or ill-conditioned")
    triangles = np.asarray(triangles_local, dtype=np.float64)
    ids = np.asarray(triangle_stable_ids)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0 or not np.isfinite(triangles).all():
        raise CertificationError("target bank triangles must be nonempty finite Fx3x3")
    if not np.issubdtype(ids.dtype, np.integer) or ids.shape != (len(triangles),):
        raise CertificationError("stable triangle IDs must be an integer vector aligned to triangles")
    if len(np.unique(ids)) != len(ids):
        raise CertificationError("target stable triangle IDs are not unique")
    if not isinstance(expected_bank_triangle_count, (int, np.integer)) or int(expected_bank_triangle_count) != len(triangles):
        raise CertificationError("provided triangles do not match the caller-pinned complete target-bank count")
    lo = _finite_array(bank_aabb_min_local, (3,), "bank AABB minimum")
    hi = _finite_array(bank_aabb_max_local, (3,), "bank AABB maximum")
    if np.any(hi <= lo):
        raise CertificationError("bank AABB must have positive extent")
    computed_lo = triangles.min(axis=(0, 1))
    computed_hi = triangles.max(axis=(0, 1))
    if not np.array_equal(lo, computed_lo) or not np.array_equal(hi, computed_hi):
        raise CertificationError("supplied bank AABB is not the exact bound of the complete provided triangle set")
    endpoint_outside = bool(np.any(end < lo) or np.any(end > hi))
    start_inside_aabb = bool(np.all(start >= lo) and np.all(start <= hi))
    segment = end - start
    segment_norm = float(np.linalg.norm(segment))
    segment_alignment = (float(np.dot(_unit(segment, "ray segment"), dlocal)) if segment_norm > 0 else -1.0)
    result: dict[str, Any] = {
        "status": "REJECT",
        "reason": None,
        "start_classification": str(start_status),
        "endpoint_outside_aabb_proven": endpoint_outside,
        "start_inside_bank_aabb": start_inside_aabb,
        "segment_source_direction_cosine": segment_alignment,
        "start_local": start.tolist(),
        "end_local": end.tolist(),
        "source_direction_local_unit": dlocal.tolist(),
        "world_start": pworld.tolist(),
        "world_to_local": M.tolist(),
        "clearance_m": float(clearance_m),
        "proper_hits": [],
        "unresolved_witnesses": [],
        "disjoint_candidate_count": 0,
        "candidate_triangle_count": 0,
        "complete_target_bank_triangle_count": int(expected_bank_triangle_count),
        "endpoint_parity": None,
    }
    if start_status != "inside":
        result["reason"] = "ray_start_not_certified_inside"
        return result
    if not start_inside_aabb:
        result["reason"] = "ray_start_outside_bank_aabb"
        return result
    if segment_norm <= 0.0 or segment_alignment < 1.0 - 1e-10:
        result["reason"] = "ray_segment_not_aligned_with_source_direction"
        return result
    if not endpoint_outside:
        result["reason"] = "ray_endpoint_not_strictly_outside_bank_aabb"
        return result
    if not math.isfinite(float(clearance_m)) or clearance_m <= 0:
        raise CertificationError("clearance must be finite and positive")
    segment_lo = np.minimum(start, end)
    segment_hi = np.maximum(start, end)
    tri_lo = triangles.min(axis=1)
    tri_hi = triangles.max(axis=1)
    candidate_mask = np.all(tri_hi >= segment_lo, axis=1) & np.all(tri_lo <= segment_hi, axis=1)
    candidate_rows = np.flatnonzero(candidate_mask)
    result["candidate_triangle_count"] = int(len(candidate_rows))
    if not callable(exact_classifier):
        raise CertificationError("caller must supply the already-pinned exact segment/triangle predicate")
    if (not isinstance(exact_classifier_sha256, str) or len(exact_classifier_sha256) != 64
            or any(c not in "0123456789abcdef" for c in exact_classifier_sha256.lower())):
        raise CertificationError("caller must supply the exact predicate source SHA-256")
    classify = exact_classifier
    result["exact_predicate_source_sha256"] = exact_classifier_sha256.lower()
    all_hits: list[tuple[Fraction, dict[str, Any]]] = []
    for row in candidate_rows:
        tri = triangles[int(row)]
        exact = classify(start, end, tri)
        classification = exact["classification"]
        if classification in ("disjoint", "parallel_disjoint"):
            result["disjoint_candidate_count"] += 1
            continue
        if classification != "proper_interior_hit":
            result["unresolved_witnesses"].append({
                "triangle_stable_id": int(ids[int(row)]),
                "classification": classification,
                "exact_result": exact,
                "triangle_row": int(row),
            })
            continue
        t_exact, bary_exact = _exact_t_barycentric(start, end, tri)
        t = float(t_exact)
        bary = np.asarray([float(v) for v in bary_exact], dtype=np.float64)
        hit_local = (1.0 - t) * start + t * end
        nlocal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nlen = float(np.linalg.norm(nlocal))
        if not math.isfinite(nlen) or nlen == 0:
            result["unresolved_witnesses"].append({
                "triangle_stable_id": int(ids[int(row)]), "classification": "invalid_degenerate_triangle",
                "triangle_row": int(row),
            })
            continue
        nunit_local = nlocal / nlen
        orient_sign_exact = _exact_dot_sign(nlocal, dlocal)
        if orient_sign_exact == 0:
            result["unresolved_witnesses"].append({
                "triangle_stable_id": int(ids[int(row)]), "classification": "tangent_or_zero_orientation",
                "triangle_row": int(row),
            })
            continue
        if orient_sign_exact < 0:
            nunit_local = -nunit_local
        nworld = _unit(M.T @ nunit_local, "world exit-plane covector")
        world_hit = pworld + np.linalg.solve(M, hit_local - start)
        hit = {
            "triangle_row": int(row),
            "triangle_stable_id": int(ids[int(row)]),
            "t_exact_numerator": str(t_exact.numerator),
            "t_exact_denominator": str(t_exact.denominator),
            "t": t,
            "barycentric": bary.tolist(),
            "hit_local": hit_local.tolist(),
            "hit_world": world_hit.tolist(),
            "unit_exit_normal_world": nworld.tolist(),
            "source_direction_dot_exit_normal": float(nworld @ np.linalg.solve(M, dlocal)),
        }
        all_hits.append((t_exact, hit))
    all_hits.sort(key=lambda item: (item[0], item[1]["triangle_stable_id"]))
    result["proper_hits"] = [h for _, h in all_hits]
    if result["unresolved_witnesses"]:
        result["reason"] = "boundary_coplanar_degenerate_or_tangent_witness"
        return result
    exact_parameters = [t for t, _ in all_hits]
    if any(a == b for a, b in zip(exact_parameters, exact_parameters[1:])):
        result["reason"] = "nonunique_exact_hit_parameter"
        result["duplicate_parameter_groups"] = [
            {"t_exact": [str(a.numerator), str(a.denominator)],
             "triangle_ids": [all_hits[i][1]["triangle_stable_id"], all_hits[i+1][1]["triangle_stable_id"]]}
            for i, (a, b) in enumerate(zip(exact_parameters, exact_parameters[1:])) if a == b
        ]
        return result
    if len(all_hits) == 0:
        result["reason"] = "inside_to_outside_ray_has_no_exact_hits"
        result["endpoint_parity"] = "even"
        return result
    odd = len(all_hits) % 2 == 1
    result["endpoint_parity"] = "odd" if odd else "even"
    if not odd:
        result["reason"] = "inside_to_outside_ray_has_even_exact_hit_parity"
        return result
    selected = all_hits[0][1]
    if float(np.dot(np.asarray(selected["unit_exit_normal_world"]), np.linalg.solve(M, dlocal))) <= 0.0:
        result["reason"] = "first_exact_hit_normal_does_not_follow_source_outward_ray"
        return result
    point = np.asarray(selected["hit_world"], dtype=np.float64)
    normal = np.asarray(selected["unit_exit_normal_world"], dtype=np.float64)
    result["selected_first_exit"] = selected
    result["plane"] = {
        "unit_normal_world": normal.tolist(),
        "point_world": point.tolist(),
        "minimum_signed_distance_m": float(normal @ point + float(clearance_m)),
        "inequality": "dot(unit_normal_world, x_world) >= minimum_signed_distance_m",
    }
    result["reason"] = "certified_first_proper_exit_from_inside_to_aabb_outside_with_odd_exact_hit_parity"
    result["status"] = "PASS"
    result["scope_limit"] = "Construction direction is source O-D prolongation; not an anatomical outward proof or whole-surface/tet clearance certificate."
    return result

