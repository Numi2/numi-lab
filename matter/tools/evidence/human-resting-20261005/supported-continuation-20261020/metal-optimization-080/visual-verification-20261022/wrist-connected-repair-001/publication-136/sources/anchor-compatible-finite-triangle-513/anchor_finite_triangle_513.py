"""Anchor-compatible fixed-triangle planes layered over frozen 466 rows.

Construction-only objective helper. Anchored pairs consider both signs of
frozen-466 triangle-SAT support axes and require every anchored corner to have
a nonpositive lower bound. Unanchored pairs retain the 466 outward policy.
This does not certify global mesh clearance.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse


def anchor_compatible_plane(skin, bone, anchored_corners, sat):
    skin = np.asarray(skin, dtype=np.float64)
    bone = np.asarray(bone, dtype=np.float64)
    if skin.shape != (3, 3) or bone.shape != (3, 3):
        raise ValueError("expected two (3,3) triangles")
    if not np.isfinite(skin).all() or not np.isfinite(bone).all():
        raise ValueError("triangle coordinates must be finite")
    anchored = tuple(sorted(set(int(c) for c in anchored_corners)))
    if any(c < 0 or c > 2 for c in anchored):
        raise ValueError("anchored corner index must be 0, 1, or 2")
    outward = sat._triangles(skin, bone)[2]
    if not anchored:
        plane = sat.triangle_sat_plane_for_pair(skin, bone)
        plane["anchor_compatible_axis_count"] = plane["candidate_axis_count"]
        plane["anchored_corner_indices"] = []
        return plane
    # For anchored pairs, test both signs of every frozen-466 triangle SAT
    # axis. The supporting plane separates the finite triangles regardless
    # of sign; only the anchor inequalities decide which side is admissible.
    candidates = []
    for base in sat.triangle_sat_axis_candidates(skin, bone):
        for sign_order, extra_sign in enumerate((1, -1)):
            cand = dict(base)
            cand["normal_world_unit"] = np.asarray(base["normal_world_unit"], dtype=np.float64) * extra_sign
            cand["axis_sign"] = int(base["axis_sign"]) * extra_sign
            cand["axis_sign_order"] = int(base["axis_sign_order"]) * 2 + sign_order
            candidates.append(cand)
    # Deduplicate coincident directions while preserving deterministic source order.
    unique = {}
    for cand in candidates:
        key = tuple(np.round(cand["normal_world_unit"], 14))
        unique.setdefault(key, cand)
    candidates = list(unique.values())
    ulp_skin = sat._float32_ulp_bound(skin)
    ulp_bone = sat._float32_ulp_bound(bone)
    feasible = []
    for cand in candidates:
        n = np.asarray(cand["normal_world_unit"], dtype=np.float64)
        fixed_proj = bone @ n
        skin_proj = skin @ n
        max_fixed = float(np.max(fixed_proj))
        epsilon = float(np.max(np.sum(np.abs(n)[None, :] * ulp_bone, axis=1))
                        + np.max(np.sum(np.abs(n)[None, :] * ulp_skin, axis=1)))
        lower = max_fixed + epsilon - skin_proj
        if anchored and any(float(lower[c]) > 0.0 for c in anchored):
            continue
        cost = max(float(max_fixed - np.min(skin_proj) + epsilon), 0.0)
        feasible.append((cost, int(cand["axis_source_order"]),
                         int(cand["axis_sign_order"]), cand, lower,
                         max_fixed, epsilon))
    if not feasible:
        raise ValueError("no signed finite-triangle SAT axis is compatible with every anchored corner")
    cost, _, _, chosen, lower, max_fixed, epsilon = min(
        feasible, key=lambda row: (row[0], row[1], row[2]))
    max_ids = np.flatnonzero(bone @ chosen["normal_world_unit"] == max_fixed)
    vertex = int(max_ids[0])
    return {
        "normal_world_unit": np.asarray(chosen["normal_world_unit"], dtype=np.float64).copy(),
        "plane_point_mm": bone[vertex].copy(),
        "float32_ulp_margin_mm": epsilon,
        "corner_signed_distance_mm": skin @ chosen["normal_world_unit"] - max_fixed,
        "corner_lower_mm": np.asarray(lower, dtype=np.float64),
        "orientation_source": "verified fixed-bank winding; selected finite-triangle support axis sign chosen to keep anchors fixed",
        "separation_claim": "all three moved skin corners meeting lower bounds separate this finite fixed-triangle pair on its support axis; axis sign may oppose local face winding",
        "axis_strategy": "triangle_sat_minimum_max_deficit",
        "axis_source": chosen["axis_source"],
        "axis_index": int(chosen["axis_index"]),
        "axis_sign": int(chosen["axis_sign"]),
        "axis_source_order": int(chosen["axis_source_order"]),
        "axis_sign_order": int(chosen["axis_sign_order"]),
        "axis_outward_dot": float(np.dot(chosen["normal_world_unit"], outward)),
        "axis_cost_mm": float(cost),
        "max_fixed_projection_mm": max_fixed,
        "min_skin_projection_mm": float(np.min(skin @ chosen["normal_world_unit"])),
        "candidate_axis_count": int(len(candidates)),
        "anchor_compatible_axis_count": int(len(feasible)),
        "anchored_corner_indices": list(anchored),
    }


def build_anchor_compatible_rows(points_mm, faces, support, anchors, pairs,
                                 bone_triangles_mm, *, sat, selected_pair_ids,
                                 total_weight=1.0):
    """Delegate row assembly to 466, replacing only selected SAT plane choices."""
    p = np.asarray(points_mm, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    anchor_set = set(map(int, np.asarray(anchors, dtype=np.int64).reshape(-1)))
    selected = {tuple(map(int, key)) for key in selected_pair_ids}
    expected = set()
    chosen = []
    for pair in pairs:
        key = (int(pair["face_row"]), int(pair["body"]),
               int(pair["bone_id"]), int(pair["triangle_id"]))
        if key in expected:
            raise ValueError(f"duplicate crossing identity {key}")
        expected.add(key)
        if key not in selected:
            continue
        nodes = f[key[0]]
        anchored_corners = [i for i, node in enumerate(nodes) if int(node) in anchor_set]
        tri_key = (key[1], key[2], key[3])
        if tri_key not in bone_triangles_mm:
            raise KeyError(f"missing fixed triangle {tri_key}")
        try:
            plane = anchor_compatible_plane(p[nodes], bone_triangles_mm[tri_key],
                                           anchored_corners, sat)
        except ValueError as exc:
            raise ValueError(f"anchor-compatible plane failed for {key}: {exc}") from exc
        chosen.append((key, p[nodes].copy(),
                       np.asarray(bone_triangles_mm[tri_key], dtype=np.float64).copy(),
                       anchored_corners, plane))
    unknown = selected - expected
    if unknown:
        raise ValueError(f"selected identities absent from crossings: {sorted(unknown)}")
    cursor = 0
    original = sat.triangle_sat_plane_for_pair
    def choose(skin_triangle_mm, bone_triangle_mm):
        nonlocal cursor
        if cursor >= len(chosen):
            raise RuntimeError("unexpected extra SAT plane selection")
        key, expected_skin, expected_bone, _corners, plane = chosen[cursor]
        if not np.array_equal(np.asarray(skin_triangle_mm), expected_skin) or not np.array_equal(np.asarray(bone_triangle_mm), expected_bone):
            raise RuntimeError(f"SAT plane selection order or triangle mismatch at {key}")
        cursor += 1
        out = dict(plane)
        out["axis_strategy"] = "triangle_sat_minimum_max_deficit"
        return out
    sat.triangle_sat_plane_for_pair = choose
    try:
        result = sat.build_fixed_bone_outward_plane_rows(
            points_mm, faces, support, anchors, pairs, bone_triangles_mm,
            total_weight=total_weight, separating_axis_pairs=selected)
    finally:
        sat.triangle_sat_plane_for_pair = original
    if cursor != len(chosen):
        raise RuntimeError(f"used {cursor} of {len(chosen)} selected SAT planes")
    result["diagnostics"]["anchor_compatible_pair_count"] = len(chosen)
    result["diagnostics"]["anchor_compatible_corner_count"] = int(
        sum(len(row[3]) for row in chosen))
    result["diagnostics"]["anchor_compatible_axes"] = [
        {"identity": list(row[0]), "axis_source": row[4]["axis_source"],
         "axis_index": row[4]["axis_index"], "axis_sign": row[4]["axis_sign"],
         "axis_outward_dot": row[4]["axis_outward_dot"],
         "axis_cost_mm": row[4]["axis_cost_mm"],
         "compatible_candidate_count": row[4]["anchor_compatible_axis_count"],
         "anchored_corner_indices": row[4]["anchored_corner_indices"]}
        for row in chosen]
    return result

