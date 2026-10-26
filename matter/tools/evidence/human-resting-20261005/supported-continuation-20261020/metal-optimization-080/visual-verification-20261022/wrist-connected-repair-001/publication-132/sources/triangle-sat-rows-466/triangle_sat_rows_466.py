"""Optional triangle-SAT-axis guard rows for fixed skin/bone triangle pairs.

This is an opt-in extension of 458's outward bone-face rows. By default it
uses the same bone-face plane and row/objective fields as 458. For selected
typed pair identities, it instead chooses deterministically among the
bone-face, skin-face, and nine edge-cross SAT axes, orienting each axis into
the verified outward bone-face hemisphere (an exactly tangent axis admits
both signs). The resulting three corner inequalities guarantee projection
separation from the whole fixed triangle when all are satisfied. They remain
local construction rows; exact whole-mesh audits are still mandatory.

The caller must supply outward-ordered fixed-bone triangles whose winding has
already been verified against their closed positive-volume bank.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

_AXIS_NAMES = ("bone_face", "skin_face", "edge_cross")


def _unit(v, eps=1e-14):
    a = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(a))
    if not np.isfinite(norm) or norm <= eps:
        return None
    return a / norm


def _float32_ulp_bound(values):
    """Return coordinatewise upward-neighbor spacing for finite float32 values."""
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("ULP inputs must be finite")
    f = np.abs(x).astype(np.float32)
    if not np.isfinite(f).all():
        raise ValueError("coordinate magnitude overflows float32")
    nxt = np.nextafter(f, np.float32(np.inf), dtype=np.float32)
    ulp = nxt.astype(np.float64) - f.astype(np.float64)
    if not np.isfinite(ulp).all() or np.any(ulp < 0):
        raise ValueError("failed to derive finite float32 ULP")
    return ulp


def _triangles(skin_triangle_mm, bone_triangle_mm):
    skin = np.asarray(skin_triangle_mm, dtype=np.float64)
    bone = np.asarray(bone_triangle_mm, dtype=np.float64)
    if skin.shape != (3, 3) or bone.shape != (3, 3):
        raise ValueError("skin and bone triangles must each have shape (3,3)")
    if not np.isfinite(skin).all() or not np.isfinite(bone).all():
        raise ValueError("triangle vertices must be finite")
    outward = _unit(np.cross(bone[1] - bone[0], bone[2] - bone[0]))
    if outward is None:
        raise ValueError("fixed bone triangle is degenerate")
    return skin, bone, outward


def outward_plane_for_pair(skin_triangle_mm, bone_triangle_mm):
    """Return 458-compatible outward bone-face rows and plane metadata."""
    skin, bone, normal = _triangles(skin_triangle_mm, bone_triangle_mm)
    # Keep 458's original bound exactly for its default outward-face path.
    ulp_q = _float32_ulp_bound(bone[0])
    ulp_x = _float32_ulp_bound(skin)
    epsilon = float(np.max(np.sum(
        np.abs(normal)[None, :] * (ulp_q[None, :] + ulp_x), axis=1)))
    signed = (skin - bone[0]) @ normal
    return {
        "normal_world_unit": normal,
        "plane_point_mm": bone[0].copy(),
        "float32_ulp_margin_mm": epsilon,
        "corner_signed_distance_mm": signed,
        "corner_lower_mm": epsilon - signed,
        "orientation_source": "ordered fixed-bank triangle vertices; caller must verify outward bank winding",
        "separation_claim": "all three moved skin corners meeting lower bounds imply strict plane separation from this fixed triangle",
    }


def triangle_sat_axis_candidates(skin_triangle_mm, bone_triangle_mm):
    """Return normalized SAT axes oriented into/outward tangent hemisphere.

    The result order is bone face, skin face, then the nine bone-edge x
    skin-edge axes, with deterministic + sign before - sign for exact tangents.
    Degenerate candidates are omitted. `axis_index` indexes a face (always
    zero) or the row-major (bone edge, skin edge) pair for edge crosses.
    """
    skin, bone, outward = _triangles(skin_triangle_mm, bone_triangle_mm)
    bone_edges = (bone[1] - bone[0], bone[2] - bone[1], bone[0] - bone[2])
    skin_edges = (skin[1] - skin[0], skin[2] - skin[1], skin[0] - skin[2])
    raw = [("bone_face", 0, outward),
           ("skin_face", 0, np.cross(skin[1] - skin[0], skin[2] - skin[0]))]
    raw.extend(("edge_cross", 3 * i + j, np.cross(be, se))
               for i, be in enumerate(bone_edges)
               for j, se in enumerate(skin_edges))
    candidates = []
    for source_order, (kind, index, vector) in enumerate(raw):
        axis = _unit(vector)
        if axis is None:
            continue
        dot = float(np.dot(axis, outward))
        # Choose the sign that has nonnegative dot. Exactly tangent axes admit
        # both signs; + is first for deterministic tie breaking.
        if dot > 0.0:
            signed_axes = (1,)
        elif dot < 0.0:
            signed_axes = (-1,)
        else:
            signed_axes = (1, -1)
        for sign_order, sign in enumerate(signed_axes):
            normal = axis * float(sign)
            outward_dot = float(np.dot(normal, outward))
            if outward_dot < 0.0:
                continue
            candidates.append({
                "normal_world_unit": normal,
                "axis_source": kind,
                "axis_index": int(index),
                "axis_sign": int(sign),
                "axis_source_order": int(source_order),
                "axis_sign_order": int(sign_order),
                "axis_outward_dot": outward_dot,
            })
    if not candidates:
        raise ValueError("triangle SAT produced no outward-hemisphere candidate axis")
    return candidates


def triangle_sat_plane_for_pair(skin_triangle_mm, bone_triangle_mm):
    """Choose the outward-hemisphere SAT axis with minimum max deficit cost."""
    skin, bone, _outward = _triangles(skin_triangle_mm, bone_triangle_mm)
    candidates = triangle_sat_axis_candidates(skin, bone)
    ulp_skin = _float32_ulp_bound(skin)
    ulp_bone = _float32_ulp_bound(bone)
    scored = []
    for candidate in candidates:
        normal = candidate["normal_world_unit"]
        bone_projection = bone @ normal
        skin_projection = skin @ normal
        max_fixed = float(np.max(bone_projection))
        min_skin = float(np.min(skin_projection))
        # Bound independent Float32 rounding of the maximizing fixed vertex
        # and minimizing skin vertex. Maxima make this valid regardless of
        # which vertex becomes extremal after rounding.
        fixed_rounding = float(np.max(np.sum(np.abs(normal)[None, :] * ulp_bone, axis=1)))
        skin_rounding = float(np.max(np.sum(np.abs(normal)[None, :] * ulp_skin, axis=1)))
        epsilon = fixed_rounding + skin_rounding
        cost = max(max_fixed - min_skin + epsilon, 0.0)
        scored.append((cost, candidate["axis_source_order"],
                       candidate["axis_sign_order"], candidate, max_fixed,
                       min_skin, epsilon))
    cost, _source_order, _sign_order, chosen, max_fixed, min_skin, epsilon = min(
        scored, key=lambda row: (row[0], row[1], row[2]))
    normal = chosen["normal_world_unit"]
    signed = skin @ normal - max_fixed
    plane_vertex = int(np.flatnonzero(bone @ normal == max_fixed)[0])
    return {
        "normal_world_unit": normal.copy(),
        "plane_point_mm": bone[plane_vertex].copy(),
        "float32_ulp_margin_mm": float(epsilon),
        "corner_signed_distance_mm": signed,
        "corner_lower_mm": max_fixed + epsilon - skin @ normal,
        "orientation_source": "ordered fixed-bank triangle outward normal; selected SAT candidate filtered to nonnegative dot",
        "separation_claim": "all three moved skin corners meeting lower bounds place the full skin triangle beyond the fixed triangle support plane on a triangle SAT candidate axis",
        "axis_strategy": "triangle_sat_minimum_max_deficit",
        "axis_source": chosen["axis_source"],
        "axis_index": chosen["axis_index"],
        "axis_sign": chosen["axis_sign"],
        "axis_outward_dot": chosen["axis_outward_dot"],
        "axis_cost_mm": float(cost),
        "max_fixed_projection_mm": float(max_fixed),
        "min_skin_projection_mm": float(min_skin),
        "candidate_axis_count": int(len(candidates)),
    }


def _validated_id_set(values, name):
    if values is None:
        return None
    result = set()
    for value in values:
        key = tuple(int(x) for x in value)
        if len(key) != 4:
            raise ValueError(f"{name} identities must be (face_row,body,bone_id,triangle_id)")
        if key in result:
            raise ValueError(f"duplicate {name} identity: {key}")
        result.add(key)
    return result


def build_fixed_bone_outward_plane_rows(
        points_mm, faces, support, anchors, pairs, bone_triangles_mm,
        total_weight=1.0, separating_axis_pairs=None):
    """Build 458-compatible rows with optional per-pair triangle-SAT axes.

    `separating_axis_pairs` is an optional iterable of typed identities
    `(face_row, body, bone_id, triangle_id)`. Only those pairs use the
    deterministic minimum-deficit SAT axis; all other pairs retain the 458
    outward bone-face plane. The return dictionary has the same fields as 458:
    `G`, `lower`, `guard_metadata`, `C`, `h`, `weights`, `metadata`, and
    `diagnostics`, with axis selection details added to metadata.
    """
    p = np.asarray(points_mm, dtype=np.float64)
    raw_faces = np.asarray(faces)
    raw_scope = np.asarray(support)
    raw_anchors = np.asarray(anchors)
    if (raw_faces.dtype.kind not in "iu" or raw_scope.dtype.kind not in "iu"
            or (raw_anchors.size and raw_anchors.dtype.kind not in "iu")):
        raise ValueError("faces and support must use integer node rows; nonempty anchors must be integer")
    f = raw_faces.astype(np.int64, copy=False)
    scope = raw_scope.astype(np.int64, copy=False)
    anchor_rows = np.asarray(raw_anchors, dtype=np.int64).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("points_mm must be finite with shape (N,3)")
    if f.ndim != 2 or f.shape[1] != 3 or np.any(f < 0) or np.any(f >= len(p)):
        raise ValueError("faces must be valid triangle node rows")
    if scope.ndim != 1 or np.any(scope < 0) or np.any(scope >= len(p)):
        raise ValueError("support contains an invalid node")
    if len(np.unique(scope)) != len(scope):
        raise ValueError("support contains duplicate nodes")
    if np.any(anchor_rows < 0) or np.any(anchor_rows >= len(p)):
        raise ValueError("anchors contain an invalid node")
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("total_weight must be finite and positive")
    sat_ids = _validated_id_set(separating_axis_pairs, "separating-axis")
    local = {int(node): i for i, node in enumerate(scope)}
    pair_list = list(pairs)
    seen = set()
    records = []
    for pair_index, pair in enumerate(pair_list):
        face = int(pair["face_row"])
        body = int(pair["body"])
        bone = int(pair["bone_id"])
        triangle = int(pair["triangle_id"])
        key = (face, body, bone, triangle)
        if key in seen:
            raise ValueError(f"duplicate crossed pair identity: {key}")
        seen.add(key)
        if face < 0 or face >= len(f):
            raise ValueError(f"crossed pair face row out of range: {face}")
        tri_key = (body, bone, triangle)
        if tri_key not in bone_triangles_mm:
            raise KeyError(f"missing fixed bone triangle {tri_key}")
        nodes = f[face]
        if any(int(node) not in local for node in nodes):
            raise ValueError(f"face {face} has a node outside support")
        if sat_ids is not None and key in sat_ids:
            plane = triangle_sat_plane_for_pair(p[nodes], bone_triangles_mm[tri_key])
        else:
            plane = outward_plane_for_pair(p[nodes], bone_triangles_mm[tri_key])
            normal = plane["normal_world_unit"]
            plane.update({
                "axis_strategy": "bone_face_458_compatible",
                "axis_source": "bone_face",
                "axis_index": 0,
                "axis_sign": 1,
                "axis_outward_dot": 1.0,
                "axis_cost_mm": float(max(
                    float(np.max(bone_triangles_mm[tri_key] @ normal)
                          - np.min(p[nodes] @ normal))
                    + plane["float32_ulp_margin_mm"], 0.0)),
            })
        records.append((pair_index, key, nodes, plane))
    if sat_ids is not None:
        unknown = sat_ids - seen
        if unknown:
            raise ValueError(f"separating-axis identities are absent from pairs: {sorted(unknown)}")

    anchor_set = set(map(int, anchor_rows))
    gr, gc, gv, lower, guard_metadata = [], [], [], [], []
    soft_records = []
    row = 0
    anchored_positive = []
    for pair_index, key, nodes, plane in records:
        normal = plane["normal_world_unit"]
        deficits = plane["corner_lower_mm"]
        active_nonanchor = [int(c) for c in range(3)
                            if deficits[c] > 1e-12 and int(nodes[c]) not in anchor_set]
        if active_nonanchor:
            soft_records.append((pair_index, key, nodes, plane, active_nonanchor))
        for corner in range(3):
            node = int(nodes[corner])
            rhs = float(deficits[corner])
            if node not in anchor_set:
                col = 3 * local[node]
                for axis in range(3):
                    if normal[axis] != 0.0:
                        gr.append(row); gc.append(col + axis); gv.append(float(normal[axis]))
            elif rhs > 1e-12:
                anchored_positive.append({
                    "pair_index": int(pair_index), "identity": list(map(int, key)),
                    "corner_index": int(corner), "node_row": node,
                    "lower_mm": rhs,
                })
            lower.append(rhs)
            sat_selected = plane["axis_strategy"] == "triangle_sat_minimum_max_deficit"
            guard_metadata.append({
                "kind": ("fixed_bone_separating_axis_corner_lower_bound" if sat_selected
                         else "fixed_bone_outward_plane_corner_lower_bound"),
                "pair_index": int(pair_index), "face_row": int(key[0]),
                "body": int(key[1]), "bone_id": int(key[2]),
                "triangle_id": int(key[3]), "corner_index": int(corner),
                "node_row": node, "normal_world_unit": normal.tolist(),
                "plane_point_mm": plane["plane_point_mm"].tolist(),
                "float32_ulp_margin_mm": plane["float32_ulp_margin_mm"],
                "lower_mm": rhs,
                "anchor_zero_coefficients": node in anchor_set,
                "axis_strategy": plane["axis_strategy"],
                "axis_source": plane["axis_source"],
                "axis_index": plane["axis_index"],
                "axis_sign": plane["axis_sign"],
                "axis_outward_dot": plane["axis_outward_dot"],
                "axis_cost_mm": plane["axis_cost_mm"],
                "max_fixed_projection_mm": plane.get("max_fixed_projection_mm"),
                "min_skin_projection_mm": plane.get("min_skin_projection_mm"),
                "candidate_axis_count": plane.get("candidate_axis_count"),
                "claim_scope": ("one_fixed_pair_projection_plane_only_not_global_clearance_certificate"
                                if not sat_selected else
                                "one_fixed_pair_triangle_SAT_projection_plane_only_not_global_clearance_certificate"),
            })
            row += 1
    G = sparse.coo_matrix((gv, (gr, gc)), shape=(row, 3 * len(scope))).tocsr()
    lower = np.asarray(lower, dtype=np.float64)

    cr, cc, cv, h, weights, metadata = [], [], [], [], [], []
    soft_pair_mass = float(total_weight) / len(soft_records) if soft_records else 0.0
    soft_row = 0
    for pair_index, key, nodes, plane, active in soft_records:
        normal = plane["normal_world_unit"]
        deficits = plane["corner_lower_mm"]
        one_weight = soft_pair_mass / len(active)
        for corner in active:
            node = int(nodes[corner])
            col = 3 * local[node]
            for axis in range(3):
                if normal[axis] != 0.0:
                    cr.append(soft_row); cc.append(col + axis); cv.append(float(normal[axis]))
            h.append(float(deficits[corner]))
            weights.append(one_weight)
            sat_selected = plane["axis_strategy"] == "triangle_sat_minimum_max_deficit"
            metadata.append({
                "kind": ("fixed_bone_separating_axis_positive_target" if sat_selected
                         else "fixed_bone_outward_plane_positive_target"),
                "pair_index": int(pair_index), "face_row": int(key[0]),
                "body": int(key[1]), "bone_id": int(key[2]),
                "triangle_id": int(key[3]), "corner_index": int(corner),
                "node_row": node, "normal_world_unit": normal.tolist(),
                "plane_point_mm": plane["plane_point_mm"].tolist(),
                "float32_ulp_margin_mm": plane["float32_ulp_margin_mm"],
                "target_mm": float(deficits[corner]),
                "normalized_weight": float(one_weight),
                "axis_strategy": plane["axis_strategy"],
                "axis_source": plane["axis_source"],
                "axis_index": plane["axis_index"],
                "axis_sign": plane["axis_sign"],
                "axis_outward_dot": plane["axis_outward_dot"],
                "axis_cost_mm": plane["axis_cost_mm"],
                "max_fixed_projection_mm": plane.get("max_fixed_projection_mm"),
                "min_skin_projection_mm": plane.get("min_skin_projection_mm"),
                "candidate_axis_count": plane.get("candidate_axis_count"),
                "claim_scope": ("soft_local_target_not_clearance_certificate" if not sat_selected
                                else "soft_local_triangle_SAT_target_not_clearance_certificate"),
            })
            soft_row += 1
    C = sparse.coo_matrix((cv, (cr, cc)), shape=(soft_row, 3 * len(scope))).tocsr()
    h = np.asarray(h, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return {
        "G": G, "lower": lower, "guard_metadata": guard_metadata,
        "C": C, "h": h, "weights": weights, "metadata": metadata,
        "diagnostics": {
            "input_pair_count": len(pair_list),
            "lower_bound_row_count": int(len(lower)),
            "soft_pair_count": len(soft_records),
            "soft_row_count": int(len(h)),
            "soft_weight_mass": float(weights.sum()),
            "anchored_positive_lower_count": len(anchored_positive),
            "anchored_positive_lowers": anchored_positive,
            "triangle_sat_axis_pair_count": int(sum(
                r[3]["axis_strategy"] == "triangle_sat_minimum_max_deficit" for r in records)),
            "construction_only": True,
        },
    }


# Additive alias clarifies that this implementation uses no physical simulation.
build_fixed_bone_separating_axis_rows_466 = build_fixed_bone_outward_plane_rows
