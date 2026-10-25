"""Offline exact-edge span preparation and geometric nearest-exit planes.

A nearest exit does not select the connected anatomical exterior. Its contact
plane is a diagnostic construction primitive, not native contact or anatomy
admission. Callers must qualify the exit side and retain full solid checks.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np


def _inside(value: str) -> bool:
    if value == "inside":
        return True
    if value in ("outside", "outside_aabb_proven", "outside_AABB_proven"):
        return False
    raise ValueError(f"unresolved endpoint classification: {value!r}")


def build_edge_span_midpoint_samples(
    witnesses,
    uncertain_witnesses,
    endpoint_classification,
    query_row_to_mechanics_node,
    mechanics_points_f32,
    *,
    first_sample_query_row=0,
):
    """Create a query sample at every nonzero exact-hit interval midpoint.

    Query-row IDs and mechanics-node IDs are intentionally separate domains.
    endpoint_classification is keyed by (mechanics_query_row, body, bone); a
    missing key means that the validated node query proved that bank AABB outside.
    Exact-equal hit parameters are coalesced; distinct parameters are preserved.
    """
    points = np.asarray(mechanics_points_f32)
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("mechanics points must be an Nx3 Float32 array")
    if not np.isfinite(points).all():
        raise ValueError("nonfinite mechanics point")
    qrow_to_node = np.asarray(query_row_to_mechanics_node)
    if qrow_to_node.ndim != 1 or not np.issubdtype(qrow_to_node.dtype, np.integer):
        raise ValueError("query-row to mechanics-node map must be a 1D integer array")
    if len(qrow_to_node) == 0 or np.any(qrow_to_node < 0) or np.any(qrow_to_node >= len(points)):
        raise ValueError("query-row to mechanics-node mapping is out of range")
    if len(set(map(int, qrow_to_node))) != len(qrow_to_node):
        raise ValueError("mechanics-node mapping must be one-to-one for this query batch")
    if uncertain_witnesses:
        first = uncertain_witnesses[0]
        raise ValueError(f"edge audit contains uncertain hit(s); refusing to skip: {first!r}")

    groups = defaultdict(list)
    for witness in witnesses:
        if witness.get("classification") != "proper_interior_hit":
            raise ValueError(f"non-proper edge hit cannot be silently skipped: {witness!r}")
        exact = witness.get("exact_result", {})
        if exact.get("classification") != "proper_interior_hit":
            raise ValueError(f"edge hit lacks proper exact result: {witness!r}")
        key = (int(witness["edge_index"]), int(witness["body"]), int(witness["bone_stable_id"]))
        groups[key].append(witness)

    samples = []
    parity_contradictions = []
    identical_t_witnesses = 0
    collapsed_sample_ids = []
    for (edge_index, body, bone), hits in sorted(groups.items()):
        hits.sort(key=lambda w: (float(w["exact_result"]["t"]), int(w["triangle_stable_id"])))
        first = hits[0]
        qrows = tuple(map(int, first["endpoint_query_rows"]))
        mechanics_nodes = tuple(map(int, first["endpoint_local_nodes"]))
        if len(qrows) != 2 or len(mechanics_nodes) != 2:
            raise ValueError(f"edge {edge_index}: malformed endpoint identity")
        mapped_nodes = tuple(int(qrow_to_node[q]) if 0 <= q < len(qrow_to_node) else -1 for q in qrows)
        if mapped_nodes != mechanics_nodes:
            raise ValueError(
                f"edge {edge_index}: endpoint query rows {qrows} map to mechanics nodes {mapped_nodes}, "
                f"but source edge declares {mechanics_nodes}"
            )
        a, b = mechanics_nodes
        if a == b or min(a, b) < 0 or max(a, b) >= len(points):
            raise ValueError(f"edge {edge_index}: invalid mechanics endpoint IDs")
        for w in hits:
            if tuple(map(int, w["endpoint_query_rows"])) != qrows or tuple(map(int, w["endpoint_local_nodes"])) != mechanics_nodes:
                raise ValueError(f"edge {edge_index}: hit endpoint identities disagree")

        cls_a = endpoint_classification.get((qrows[0], body, bone), "outside_aabb_proven")
        cls_b = endpoint_classification.get((qrows[1], body, bone), "outside_aabb_proven")
        inside = _inside(cls_a)
        end_inside_by_parity = inside
        unique_hits = []
        for w in hits:
            t = float(w["exact_result"]["t"])
            if not math.isfinite(t) or not 0.0 < t < 1.0:
                raise ValueError(f"edge {edge_index}: invalid exact hit parameter {t!r}")
            bary = np.asarray(w["exact_result"].get("barycentric", []), dtype=np.float64)
            if bary.shape != (3,) or not np.isfinite(bary).all() or abs(float(bary.sum()) - 1.0) > 1e-10 or np.min(bary) < -1e-10 or np.max(bary) > 1.0 + 1e-10:
                raise ValueError(f"edge {edge_index}: invalid exact hit barycentrics {bary.tolist()}")
            if unique_hits and t == unique_hits[-1][0]:
                unique_hits[-1][1].append(w)
                identical_t_witnesses += 1
            else:
                unique_hits.append([t, [w]])
        end_inside_by_parity ^= bool(len(unique_hits) % 2)
        if end_inside_by_parity != _inside(cls_b):
            parity_contradictions.append({
                "edge_index": edge_index, "body": body, "bone_stable_id": bone,
                "endpoint_query_rows": list(qrows), "endpoint_mechanics_nodes": list(mechanics_nodes),
                "endpoint_classifications": [cls_a, cls_b], "unique_hit_count": len(unique_hits),
                "hit_t": [h[0] for h in unique_hits],
            })

        boundaries = [0.0, *[h[0] for h in unique_hits], 1.0]
        hit_groups = [None, *[h[1] for h in unique_hits], None]
        expected_inside = inside
        point_a = points[a].astype(np.float64)
        point_b = points[b].astype(np.float64)
        seen_bits = set()
        for span_index, (t0, t1) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if not t1 > t0:
                raise ValueError(f"edge {edge_index}: zero-width exact hit span")
            tm = 0.5 * (t0 + t1)
            p64 = (1.0 - tm) * point_a + tm * point_b
            p32 = p64.astype(np.float32)
            residual = p32.astype(np.float64) - p64
            ulp = np.abs(np.spacing(p32))
            limit = 0.5 * ulp.astype(np.float64) + np.finfo(np.float32).smallest_subnormal
            if np.any(np.abs(residual) > limit):
                raise ValueError(f"edge {edge_index}: sample Float32 rounding is not within half ULP")
            pbits = tuple(map(int, p32.view(np.uint32)))
            a_bits = tuple(map(int, points[a].view(np.uint32)))
            b_bits = tuple(map(int, points[b].view(np.uint32)))
            lower_hits = hit_groups[span_index]
            upper_hits = hit_groups[span_index + 1]
            lower_hit_bits = None
            upper_hit_bits = None
            if lower_hits is not None:
                lower_hit_bits = tuple(map(int, ((1.0-boundaries[span_index])*point_a + boundaries[span_index]*point_b).astype(np.float32).view(np.uint32)))
            if upper_hits is not None:
                upper_hit_bits = tuple(map(int, ((1.0-boundaries[span_index+1])*point_a + boundaries[span_index+1]*point_b).astype(np.float32).view(np.uint32)))
            collapsed = (pbits == a_bits or pbits == b_bits or pbits in seen_bits or
                         pbits == lower_hit_bits or pbits == upper_hit_bits)
            sample_row = int(first_sample_query_row) + len(samples)
            sample = {
                "sample_id": len(samples), "sample_query_row": sample_row,
                "source_domain": "mechanics_node_row", "edge_index": edge_index,
                "mechanics_node_a": a, "mechanics_node_b": b,
                "endpoint_query_rows": list(qrows), "body": body, "bone_stable_id": bone,
                "span_index": span_index, "span_t0": t0, "span_t1": t1, "t_mid": tm,
                "expected_inside_by_exact_parity": bool(expected_inside),
                "point_world_f32": [float(v) for v in p32], "point_world_f32_bits": list(pbits),
                "edge_interpolation_residual_m": [float(v) for v in residual],
                "edge_interpolation_half_ulp_limit_m": [float(v) for v in limit],
                "f32_collapsed_span": bool(collapsed),
                "lower_boundary_hit_f32_bits": lower_hit_bits,
                "upper_boundary_hit_f32_bits": upper_hit_bits,
                "lower_boundary_hit_triangles": [] if lower_hits is None else [int(w["triangle_stable_id"]) for w in lower_hits],
                "upper_boundary_hit_triangles": [] if upper_hits is None else [int(w["triangle_stable_id"]) for w in upper_hits],
            }
            if collapsed:
                collapsed_sample_ids.append(sample_row)
            samples.append(sample)
            seen_bits.add(pbits)
            expected_inside = not expected_inside

    for i, sample in enumerate(samples):
        if sample["sample_query_row"] != int(first_sample_query_row) + i:
            raise AssertionError("sample_query_row is not a contiguous sample-query domain")
    return samples, {
        "edge_bank_groups": len(groups), "span_midpoint_samples": len(samples),
        "endpoint_parity_contradiction_count": len(parity_contradictions),
        "endpoint_parity_contradictions": parity_contradictions,
        "identical_t_duplicate_witnesses_coalesced": identical_t_witnesses,
        "f32_collapsed_sample_query_rows": collapsed_sample_ids,
        "mechanics_query_row_mapping_count": int(len(qrow_to_node)),
        "sample_query_row_domain": "separate from mechanics_node_row; each sample maps to edge endpoint nodes",
    }

def validate_sample_query_rows(samples, queried_points_f32, node_rows, pair_rows, *, expected_bank_count=185, aabb_candidate_keys=None):
    """Fail closed on row-domain, output partition, ambiguity, or missing queries."""
    points = np.asarray(queried_points_f32)
    if points.dtype != np.float32 or points.ndim != 2 or points.shape != (len(samples), 3):
        raise ValueError("sample query points do not match the sample table")
    qmap = {}
    for i, sample in enumerate(samples):
        if int(sample["sample_query_row"]) != i:
            raise ValueError("sample_query_row must index the separate sample query batch")
        expected_bits = np.asarray(sample["point_world_f32_bits"], dtype=np.uint32)
        if not np.array_equal(points[i].view(np.uint32), expected_bits):
            raise ValueError(f"sample {i}: input Float32 point bits do not match manifest")
        if sample.get("source_domain") != "mechanics_node_row":
            raise ValueError(f"sample {i}: missing explicit mechanics source domain")
    if len(node_rows) != len(samples):
        raise ValueError(f"sample query returned {len(node_rows)} node rows for {len(samples)} samples")
    seen_rows = set()
    for row in node_rows:
        qrow = int(row["query_row"])
        if qrow in seen_rows or not 0 <= qrow < len(samples):
            raise ValueError(f"duplicate or out-of-domain sample query row {qrow}")
        seen_rows.add(qrow)
        counts = [int(row[k]) for k in ("aabb_proven_outside_banks", "parity_outside_banks", "inside_banks", "ambiguous_banks", "invalid_banks")]
        if int(row["total_bone_banks"]) != expected_bank_count or sum(counts) != expected_bank_count:
            raise ValueError(f"sample query row {qrow}: bank partition is incomplete")
        if counts[3] or counts[4]:
            raise ValueError(f"sample query row {qrow}: ambiguous/invalid bank classification")
    if len(seen_rows) != len(samples):
        raise ValueError("sample query row IDs are incomplete")
    candidate_keys = set(aabb_candidate_keys or ())
    for row in pair_rows:
        qrow = int(row["query_row"])
        key = (qrow, int(row["bank_body"]), int(row["bank_stable_id"]))
        if key in qmap or not 0 <= qrow < len(samples):
            raise ValueError(f"duplicate or out-of-domain sample pair key {key}")
        if int(row["ambiguous"]) != 0 or row["classification"] not in ("inside", "outside"):
            raise ValueError(f"sample pair has unresolved classification: {row!r}")
        qmap[key] = row
    target_status = {}
    for sample in samples:
        key = (int(sample["sample_query_row"]), int(sample["body"]), int(sample["bone_stable_id"]))
        if key in qmap:
            target_status[int(sample["sample_id"])] = qmap[key]["classification"]
        elif key in candidate_keys:
            raise ValueError(f"AABB candidate target bank query is missing: {key}")
        else:
            target_status[int(sample["sample_id"])] = "outside_aabb_proven"
    return qmap, target_status

def closest_point_barycentric(point, triangle, *, tolerance_m=5e-9):
    """Recover and validate the barycentric identity of a queried closest point."""
    p = np.asarray(point, dtype=np.float64)
    tri = np.asarray(triangle, dtype=np.float64)
    if p.shape != (3,) or tri.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(tri).all():
        raise ValueError("closest-point inputs must be finite 3D coordinates")
    u = tri[1] - tri[0]
    v = tri[2] - tri[0]
    gram = np.array([[u @ u, u @ v], [u @ v, v @ v]], dtype=np.float64)
    det = float(np.linalg.det(gram))
    if not math.isfinite(det) or det <= np.finfo(np.float64).tiny:
        raise ValueError("nearest-exit source triangle is degenerate")
    rhs = np.array([u @ (p - tri[0]), v @ (p - tri[0])], dtype=np.float64)
    uv = np.linalg.solve(gram, rhs)
    bary = np.array([1.0 - uv.sum(), uv[0], uv[1]], dtype=np.float64)
    reconstructed = bary @ tri
    residual = float(np.linalg.norm(reconstructed - p))
    if residual > tolerance_m or np.min(bary) < -1e-7 or np.max(bary) > 1.0 + 1e-7 or abs(float(bary.sum()) - 1.0) > 1e-12:
        raise ValueError(f"nearest-exit closest point has invalid barycentric identity: residual={residual}, bary={bary.tolist()}")
    return bary, residual


def nearest_exit_contact(sample, pair_row, triangle_local, world_to_local, local_offset, body_position, *, node_count, clearance_m=5e-5):
    """Build a two-node edge interpolation plane from the actual inside midpoint query."""
    if int(pair_row["query_row"]) != int(sample["sample_query_row"]):
        raise ValueError("sample query row does not match pair row")
    body = int(sample["body"])
    bone = int(sample["bone_stable_id"])
    if int(pair_row["bank_body"]) != body or int(pair_row["bank_stable_id"]) != bone:
        raise ValueError("queried bank identity does not match tagged edge span")
    if pair_row["classification"] != "inside" or int(pair_row["ambiguous"]) != 0:
        raise ValueError("nearest-exit plane requires a uniquely classified inside midpoint")
    if int(pair_row["nearest_bone"]) != bone:
        raise ValueError("nearest bone identity differs from tagged bank")
    tri = np.asarray(triangle_local, dtype=np.float64)
    qlocal = np.asarray([float(pair_row[f"query_local_{c}"]) for c in "xyz"], dtype=np.float64)
    hlocal = np.asarray([float(pair_row[f"closest_{c}"]) for c in "xyz"], dtype=np.float64)
    bary, bary_residual = closest_point_barycentric(hlocal, tri)
    normal_local = hlocal - qlocal
    norm = float(np.linalg.norm(normal_local))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("inside midpoint has no nonzero nearest-exit direction")
    normal_local /= norm
    rot = np.asarray(world_to_local, dtype=np.float64)
    offset = np.asarray(local_offset, dtype=np.float64)
    pos = np.asarray(body_position, dtype=np.float64)
    if rot.shape != (3, 3) or offset.shape != (3,) or pos.shape != (3,):
        raise ValueError("body transform has invalid shape")
    normal_world = rot.T @ normal_local
    normal_norm = float(np.linalg.norm(normal_world))
    if not math.isfinite(normal_norm) or normal_norm <= 1e-12:
        raise ValueError("invalid transformed nearest-exit covector")
    normal_world /= normal_norm
    hworld = pos + np.linalg.solve(rot, hlocal + offset)
    a = int(sample["mechanics_node_a"])
    b = int(sample["mechanics_node_b"])
    tm = float(sample["t_mid"])
    if a == b or not (0.0 < tm < 1.0) or min(a, b) < 0 or max(a, b) >= node_count:
        raise ValueError("edge sample has invalid mechanics-node support")
    weights = (1.0 - tm, tm)
    coefficients = [
        {"node": a, "weight": weights[0], "normal_world": normal_world.tolist()},
        {"node": b, "weight": weights[1], "normal_world": normal_world.tolist()},
    ]
    target = float(normal_world @ hworld + clearance_m)
    return {
        "coefficients": coefficients,
        "target_m": target,
        "nearest_exit_world_normal": normal_world.tolist(),
        "closest_world_m": hworld.tolist(),
        "closest_triangle_barycentric": bary.tolist(),
        "closest_barycentric_residual_m": bary_residual,
        "sample_query_row": int(sample["sample_query_row"]),
        "source_edge_index": int(sample["edge_index"]),
        "mechanics_node_rows": [a, b],
        "t_mid": tm,
        "body": body,
        "bone_stable_id": bone,
        "nearest_triangle_stable_id": int(pair_row["nearest_triangle"]),
        "clearance_m": float(clearance_m),
    }

def validate_sample_points_csv(points_csv, samples, points_f32):
    """Verify exported query rows preserve each sample's F32 coordinates and bits."""
    import csv
    points = np.asarray(points_f32)
    if points.dtype != np.float32 or points.shape != (len(samples), 3):
        raise ValueError("sample points CSV input has the wrong F32 shape")
    with open(points_csv, newline="") as f:
        reader = csv.DictReader(f)
        expected_fields = ["query_row", "x_m", "y_m", "z_m", "x_bits_u32", "y_bits_u32", "z_bits_u32"]
        if reader.fieldnames != expected_fields:
            raise ValueError(f"sample query CSV header mismatch: {reader.fieldnames!r}")
        rows = list(reader)
    if len(rows) != len(samples):
        raise ValueError("sample query CSV row count differs from sample mapping")
    max_coordinate_error = 0.0
    for i, (row, sample) in enumerate(zip(rows, samples)):
        if int(row["query_row"]) != int(sample["sample_query_row"]) or int(row["query_row"]) != i:
            raise ValueError(f"sample query CSV row {i} is mapped to the wrong sample_query_row")
        emitted_bits = np.asarray([int(row[k]) for k in ("x_bits_u32", "y_bits_u32", "z_bits_u32")], dtype=np.uint32)
        expected_bits = points[i].view(np.uint32)
        manifest_bits = np.asarray(sample["point_world_f32_bits"], dtype=np.uint32)
        if not np.array_equal(emitted_bits, expected_bits) or not np.array_equal(emitted_bits, manifest_bits):
            raise ValueError(f"sample query CSV row {i} changed the submitted Float32 identity")
        emitted_values = np.asarray([float(row[k]) for k in ("x_m", "y_m", "z_m")], dtype=np.float64).astype(np.float32)
        if not np.array_equal(emitted_values.view(np.uint32), expected_bits):
            raise ValueError(f"sample query CSV decimal coordinates do not round back to the submitted Float32 point at row {i}")
        max_coordinate_error = max(max_coordinate_error, float(np.max(np.abs(emitted_values.astype(np.float64) - points[i].astype(np.float64)))))
    return {"rows": len(rows), "max_decimal_roundtrip_error_m": max_coordinate_error, "bitwise_sample_identity": True}


# Connected source-direction exits; nearest-exit diagnostics above remain separate.

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


def build_reference_outward_field(
    reference_points_m, *, source_global_node_ids, output_source_input_row,
    native_parent_tetrahedra, generated_output_rows, generated_parent_tet_rows,
    generated_parent_barycentric, column_count=2411, direction_tolerance=1e-8,
    strict_parent_agreement=True, reference_ancestry_tolerance_m=1e-10,
):
    """Derive layer directions from the same reference as the mechanical mesh.

    Identity labels never index layer coordinates. Direct source-input rows
    invert the compact output map; all generated reference positions must match
    their declared native-parent barycentric construction in this same frame.
    """
    reference = np.asarray(reference_points_m)
    rows = np.asarray(output_source_input_row)
    labels = np.asarray(source_global_node_ids)
    count = int(column_count)
    if count <= 0 or count != column_count or labels.shape != (3*count,):
        raise CertificationError("complete three-layer source domain required")
    if reference.ndim != 2 or reference.shape[1] != 3 or reference.dtype.kind not in "fiu" or not np.isfinite(reference).all():
        raise CertificationError("finite mechanical reference Nx3 coordinates required")
    if rows.shape != (len(reference),) or rows.dtype.kind not in "iu" or np.any(rows < -1) or np.any(rows >= len(labels)):
        raise CertificationError("reference source-input row domain is invalid")
    direct = np.flatnonzero(rows >= 0)
    if len(direct) != len(labels) or not np.array_equal(np.sort(rows[direct]),np.arange(len(labels))):
        raise CertificationError("reference must cover each source-input row exactly once")
    tolerance = float(reference_ancestry_tolerance_m)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise CertificationError("positive finite reference ancestry tolerance required")
    source_reference = np.empty((len(labels),3),dtype=np.float64)
    source_reference[rows[direct]] = reference[direct]
    # Existing owner performs all typed ancestry and field-domain checks.
    field, report = build_source_outward_field(
        source_global_node_ids=labels, output_source_input_row=rows,
        outer_points_by_column_m=source_reference[:count],
        deep_points_by_column_m=source_reference[2*count:3*count],
        native_parent_tetrahedra=native_parent_tetrahedra,
        generated_output_rows=generated_output_rows,
        generated_parent_tet_rows=generated_parent_tet_rows,
        generated_parent_barycentric=generated_parent_barycentric,
        column_count=count, direction_tolerance=direction_tolerance,
        strict_parent_agreement=strict_parent_agreement)
    generated = np.asarray(generated_output_rows,dtype=np.int64)
    parents = np.asarray(generated_parent_tet_rows,dtype=np.int64)
    bary = np.asarray(generated_parent_barycentric,dtype=np.float64)
    tets = np.asarray(native_parent_tetrahedra,dtype=np.int64)
    predicted = np.einsum("ij,ijk->ik",bary,source_reference[tets[parents]])
    residual = np.linalg.norm(predicted-reference[generated],axis=1)
    maximum = float(residual.max()) if len(residual) else 0.
    if maximum > tolerance:
        witness = int(np.argmax(residual))
        raise CertificationError(
            f"generated reference row {int(generated[witness])} disagrees with declared ancestry: {maximum:.9g} m")
    report.update({
        "direction_coordinate_frame":"mechanical reference_points_m via explicit source-input row inverse",
        "legacy_layer_coordinate_arrays_used":False,
        "generated_reference_ancestry_max_residual_m":maximum,
        "reference_ancestry_tolerance_m":tolerance,
    })
    return field, report
