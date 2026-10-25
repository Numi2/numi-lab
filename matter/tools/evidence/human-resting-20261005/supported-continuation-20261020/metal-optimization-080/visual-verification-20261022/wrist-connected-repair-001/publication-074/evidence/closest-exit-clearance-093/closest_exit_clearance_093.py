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
