"""Opt-in per-corner projected deficits for validated 335 face targets.

Rows are optimization targets only; they certify neither endpoints nor
whole-triangle separation from any bone bank.
"""
from collections import Counter
from functools import wraps
import numpy as np
from scipy import sparse


def build_whole_face_clearance_rows(points_mm, faces, support, anchors,
                                    point_targets, face_targets, bary_owner,
                                    original_builder):
    # Reuse frozen 408 assembly and validation, retaining its point rows exactly.
    old_C, old_h, old_w, old_meta = original_builder(
        points_mm, faces, support, anchors, point_targets, face_targets, bary_owner)
    point_rows = np.asarray(point_targets["point_rows"], dtype=np.int64)
    rows = np.asarray(face_targets["face_rows"], dtype=np.int64)
    nodes = np.asarray(face_targets["face_nodes"], dtype=np.int64)
    bary = np.asarray(face_targets["barycentric_weights"], dtype=np.float64)
    samples = np.asarray(face_targets["face_samples_m"], dtype=np.float64)
    endpoints = np.asarray(face_targets["face_targets_f32_m"], dtype=np.float32)
    directions = np.asarray(face_targets["face_directions"], dtype=np.float64)
    p = np.asarray(points_mm, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    scope = np.asarray(support, dtype=np.int64)
    anchor_set = set(map(int, np.asarray(anchors, dtype=np.int64)))
    k = len(rows)
    if (nodes.shape != (k, 3) or bary.shape != (k, 3) or samples.shape != (k, 3)
            or endpoints.shape != (k, 3) or directions.shape != (k, 3)):
        raise ValueError("face-target arrays have inconsistent shapes")
    if np.any(rows < 0) or np.any(rows >= len(f)) or not np.array_equal(f[rows], nodes):
        raise ValueError("face target identity/order does not match candidate faces")
    if (not np.isfinite(p).all() or not np.isfinite(bary).all()
            or not np.isfinite(endpoints).all() or not np.isfinite(directions).all()):
        raise ValueError("face target inputs contain nonfinite values")
    if k and np.max(np.abs(np.linalg.norm(directions, axis=1) - 1.0)) > 2e-10:
        raise ValueError("face target directions are not unit world vectors")
    npoint = len(point_rows)
    if (old_C.shape[0] != npoint + k or len(old_h) != npoint + k
            or len(old_w) != npoint + k or len(old_meta) != npoint + k):
        raise ValueError("original 408 builder returned unexpected row counts")
    point_C = sparse.csr_matrix(old_C[:npoint], dtype=np.float64)
    point_h = np.asarray(old_h[:npoint], dtype=np.float64).copy()
    point_w = np.asarray(old_w[:npoint], dtype=np.float64).copy()
    point_meta = [dict(x) for x in old_meta[:npoint]]
    local = {int(node): i for i, node in enumerate(scope)}
    if any(int(node) not in local for node in point_rows) or any(int(node) not in local for node in nodes.ravel()):
        raise ValueError("target node lies outside expanded support")

    target_mm = endpoints.astype(np.float64) * 1000.0
    deficits = (np.einsum("ij,ikj->ik", directions,
                          target_mm[:, None, :] - p[nodes])
                if k else np.empty((0, 3), dtype=np.float64))
    positive = [np.flatnonzero(deficits[i] > 0.0) for i in range(k)]
    active_pairs = [i for i, corners in enumerate(positive) if len(corners)]
    active_faces = sorted(set(int(rows[i]) for i in active_pairs))
    pair_count = Counter(int(rows[i]) for i in active_pairs)
    face_mass = (0.5 if npoint else 1.0) / len(active_faces) if active_faces else 0.0

    archive = face_targets.get("face_npz", {})
    def bank_array(name):
        a = np.asarray(archive.get(name, np.full(k, -1)), dtype=np.int64)
        if a.shape != (k,):
            raise ValueError(f"face target identity array {name} has wrong shape")
        return a
    bodies = bank_array("bank_body")
    bones = bank_array("bank_stable_bone_id")
    triangles = bank_array("bank_triangle_stable_id")

    out_nodes, out_dirs, out_h, face_meta = [], [], [], []
    for i in active_pairs:
        corners = positive[i]
        per_corner = face_mass / pair_count[int(rows[i])] / len(corners)
        for corner in corners:
            node = int(nodes[i, corner])
            out_nodes.append(node)
            out_dirs.append(directions[i])
            out_h.append(float(deficits[i, corner]))
            face_meta.append({
                "kind": "face", "face_id": int(rows[i]), "source_index": int(i),
                "bank_body": int(bodies[i]), "bank_stable_bone_id": int(bones[i]),
                "bank_triangle_stable_id": int(triangles[i]),
                "corner_index": int(corner), "corner_row": node,
                "target_mm": float(deficits[i, corner]),
                "target_plane_endpoint_f32_m": endpoints[i].astype(float).tolist(),
                "proposal_direction_world_unit": directions[i].tolist(),
                "normalized_weight": float(per_corner),
                "source_face_target_record_count": int(k),
                "active_face_target_record_count": int(len(active_pairs)),
                "skipped_face_target_record_count": int(k - len(active_pairs)),
                "classification": "optimization_row_only_no_endpoint_or_whole_triangle_certificate",
            })

    rr, cc, vv = [], [], []
    for ri, (node, direction) in enumerate(zip(out_nodes, out_dirs)):
        if node in anchor_set:
            continue
        col = 3 * local[node]
        for axis, value in enumerate(direction):
            if value:
                rr.append(ri); cc.append(col + axis); vv.append(float(value))
    face_C = sparse.coo_matrix((vv, (rr, cc)),
                               shape=(len(out_h), 3 * len(scope))).tocsr()
    face_h = np.asarray(out_h, dtype=np.float64)
    face_w = np.asarray([x["normalized_weight"] for x in face_meta], dtype=np.float64)

    if npoint:
        wanted = 0.5 if len(face_h) else 1.0
        mass = float(point_w.sum())
        if mass <= 0.0:
            raise ValueError("original 408 point weights have no positive mass")
        point_w *= wanted / mass
        for record, weight in zip(point_meta, point_w):
            record["normalized_weight"] = float(weight)
    C = sparse.vstack((point_C, face_C), format="csr")
    h = np.concatenate((point_h, face_h))
    weights = np.concatenate((point_w, face_w))
    meta = point_meta + face_meta
    if len(weights) and (not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("objective weights must be finite and positive")
    return C, h, weights, meta


def replacement_for(original_builder):
    """Capture the frozen 408 function and return a drop-in opt-in replacement."""
    @wraps(original_builder)
    def replacement(points_mm, faces, support, anchors, point_targets, face_targets, bary_owner):
        return build_whole_face_clearance_rows(
            points_mm, faces, support, anchors, point_targets, face_targets,
            bary_owner, original_builder)
    replacement.__name__ = "build_clearance_rows_whole_face_objective_429"
    return replacement
