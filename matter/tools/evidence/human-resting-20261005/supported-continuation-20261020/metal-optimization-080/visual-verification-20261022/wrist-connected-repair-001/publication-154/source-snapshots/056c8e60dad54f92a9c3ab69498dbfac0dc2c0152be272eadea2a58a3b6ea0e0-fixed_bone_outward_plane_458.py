"""Experimental fixed-bone outward-plane rows for current skin/bone crossings.

The caller must supply actual emitted fixed-bone triangle vertices in world mm
with their oriented vertex order verified outward for the closed positive-volume
bank. Three corner lower bounds separate one skin triangle from that fixed
bone triangle when all are met. These local rows are not a global clearance
certificate; complete exact audits remain required after every fit.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


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
    ulp = (nxt.astype(np.float64) - f.astype(np.float64))
    if not np.isfinite(ulp).all() or np.any(ulp < 0):
        raise ValueError("failed to derive finite float32 ULP")
    return ulp


def outward_plane_for_pair(skin_triangle_mm, bone_triangle_mm):
    """Derive the oriented bone-face plane from ordered world-mm vertices."""
    skin = np.asarray(skin_triangle_mm, dtype=np.float64)
    bone = np.asarray(bone_triangle_mm, dtype=np.float64)
    if skin.shape != (3, 3) or bone.shape != (3, 3):
        raise ValueError("skin and bone triangles must each have shape (3,3)")
    if not np.isfinite(skin).all() or not np.isfinite(bone).all():
        raise ValueError("triangle vertices must be finite")
    normal = _unit(np.cross(bone[1] - bone[0], bone[2] - bone[0]))
    if normal is None:
        raise ValueError("fixed bone triangle is degenerate")
    # One ULP per participating coordinate bounds the total displacement
    # caused by independently rounding q0 and a skin corner to float32.
    ulp_q = _float32_ulp_bound(bone[0])
    ulp_x = _float32_ulp_bound(skin)
    epsilon = float(np.max(np.sum(np.abs(normal)[None, :] * (ulp_q[None, :] + ulp_x), axis=1)))
    signed = (skin - bone[0]) @ normal
    deficits = epsilon - signed
    return {
        "normal_world_unit": normal,
        "plane_point_mm": bone[0].copy(),
        "float32_ulp_margin_mm": epsilon,
        "corner_signed_distance_mm": signed,
        "corner_lower_mm": deficits,
        "orientation_source": "ordered fixed-bank triangle vertices; caller must verify outward bank winding",
        "separation_claim": "all three moved skin corners meeting lower bounds imply strict plane separation from this fixed triangle",
    }


def build_fixed_bone_outward_plane_rows(
        points_mm, faces, support, anchors, pairs, bone_triangles_mm,
        total_weight=1.0):
    """Build three lower-bound rows and a positive-only soft objective subset.

    pairs is a sequence of mappings with face_row, body, bone_id, triangle_id.
    bone_triangles_mm maps (body,bone_id,triangle_id) to the actual emitted,
    outward-ordered world-mm triangle. Returns a dictionary containing:
      G, lower, guard_metadata: all 3 per-pair lower-bound inequalities;
      C, h, weights, metadata: positive normalized LP slack target rows only;
      diagnostics: skipped/anchored conflict counts.

    In displacement coordinates d, G d >= lower. Sparse columns correspond to
    local displacement components ordered xyz for each support node. Anchors
    have zero coefficients in both matrices. Positive target rows alone avoid
    pulling already-clear corners inward through an equality objective.
    """
    p = np.asarray(points_mm, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    scope = np.asarray(support, dtype=np.int64)
    anchor_set = set(map(int, np.asarray(anchors, dtype=np.int64)))
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("points_mm must be finite with shape (N,3)")
    if f.ndim != 2 or f.shape[1] != 3 or np.any(f < 0) or np.any(f >= len(p)):
        raise ValueError("faces must be valid triangle node rows")
    if scope.ndim != 1 or np.any(scope < 0) or np.any(scope >= len(p)):
        raise ValueError("support contains an invalid node")
    if len(np.unique(scope)) != len(scope):
        raise ValueError("support contains duplicate nodes")
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("total_weight must be finite and positive")
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
        plane = outward_plane_for_pair(p[nodes], bone_triangles_mm[tri_key])
        records.append((pair_index, key, nodes, plane))

    # Guard block contains 3 rows per input pair, including negative lower
    # bounds that preserve the selected halfspace in a hard-inequality solver.
    gr, gc, gv, lower, guard_metadata = [], [], [], [], []
    soft_records = []
    row = 0
    anchored_positive = []
    for pair_index, key, nodes, plane in records:
        normal = plane["normal_world_unit"]
        deficits = plane["corner_lower_mm"]
        active_nonanchor = [
            int(c) for c in range(3)
            if deficits[c] > 1e-12 and int(nodes[c]) not in anchor_set
        ]
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
            guard_metadata.append({
                "kind": "fixed_bone_outward_plane_corner_lower_bound",
                "pair_index": int(pair_index), "face_row": int(key[0]),
                "body": int(key[1]), "bone_id": int(key[2]),
                "triangle_id": int(key[3]), "corner_index": int(corner),
                "node_row": node, "normal_world_unit": normal.tolist(),
                "plane_point_mm": plane["plane_point_mm"].tolist(),
                "float32_ulp_margin_mm": plane["float32_ulp_margin_mm"],
                "lower_mm": rhs,
                "anchor_zero_coefficients": node in anchor_set,
                "claim_scope": "one_fixed_pair_plane_only_not_global_clearance_certificate",
            })
            row += 1
    G = sparse.coo_matrix((gv, (gr, gc)), shape=(row, 3 * len(scope))).tocsr()
    lower = np.asarray(lower, dtype=np.float64)

    # Equal pair mass; positive, movable corner targets divide their pair mass.
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
            metadata.append({
                "kind": "fixed_bone_outward_plane_positive_target",
                "pair_index": int(pair_index), "face_row": int(key[0]),
                "body": int(key[1]), "bone_id": int(key[2]),
                "triangle_id": int(key[3]), "corner_index": int(corner),
                "node_row": node, "normal_world_unit": normal.tolist(),
                "plane_point_mm": plane["plane_point_mm"].tolist(),
                "float32_ulp_margin_mm": plane["float32_ulp_margin_mm"],
                "target_mm": float(deficits[corner]),
                "normalized_weight": float(one_weight),
                "claim_scope": "soft_local_target_not_clearance_certificate",
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
            "construction_only": True,
        },
    }

