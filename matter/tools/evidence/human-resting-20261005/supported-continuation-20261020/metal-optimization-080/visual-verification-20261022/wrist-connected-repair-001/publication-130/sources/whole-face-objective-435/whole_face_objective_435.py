"""Opt-in grouped exit-plane objective for certified 335 construction targets.

Select one existing endpoint plane per (skin face, body, stable bone ID), then
emit positive per-corner projection rows for that plane. This objective does
not certify a triangle, a whole surface, or a complete bank outside state.
"""
from collections import defaultdict
from pathlib import Path
import importlib.util
import numpy as np
from scipy import sparse


def _load_429():
    path = Path(__file__).resolve().parent.parent / "whole-face-objective-429" / "whole_face_objective_429.py"
    spec = importlib.util.spec_from_file_location("_whole_face_objective_429_for_435", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load 429 row builder at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_one_plane_per_face_bone(points_mm, face_targets):
    """Return one stable 335 source index per face/body/bone group.

    The selected certified plane minimizes its largest positive corner
    deficit. Exact ties choose the lower original source index.
    """
    p = np.asarray(points_mm, dtype=np.float64)
    rows = np.asarray(face_targets["face_rows"], dtype=np.int64)
    nodes = np.asarray(face_targets["face_nodes"], dtype=np.int64)
    targets = np.asarray(face_targets["face_targets_f32_m"], dtype=np.float32)
    dirs = np.asarray(face_targets["face_directions"], dtype=np.float64)
    archive = face_targets.get("face_npz", {})
    bodies = np.asarray(archive.get("bank_body"), dtype=np.int64)
    bones = np.asarray(archive.get("bank_stable_bone_id"), dtype=np.int64)
    k = len(rows)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("points_mm must be finite with shape (N,3)")
    if (nodes.shape != (k, 3) or targets.shape != (k, 3) or dirs.shape != (k, 3)
            or bodies.shape != (k,) or bones.shape != (k,)):
        raise ValueError("face target identities and coordinate arrays have inconsistent shapes")
    if np.any(nodes < 0) or np.any(nodes >= len(p)):
        raise ValueError("face target node lies outside points")
    if not np.isfinite(targets).all() or not np.isfinite(dirs).all():
        raise ValueError("face target arrays contain nonfinite values")
    if k and np.max(np.abs(np.linalg.norm(dirs, axis=1) - 1.0)) > 2e-10:
        raise ValueError("face target directions must be unit vectors")
    target_mm = targets.astype(np.float64) * 1000.0
    deficits = (np.einsum("ij,ikj->ik", dirs, target_mm[:, None, :] - p[nodes])
                if k else np.empty((0, 3), dtype=np.float64))
    scores = np.max(np.maximum(deficits, 0.0), axis=1) if k else np.empty(0)
    groups = defaultdict(list)
    for i in range(k):
        groups[(int(rows[i]), int(bodies[i]), int(bones[i]))].append(i)
    winners = {}
    for _, indices in sorted(groups.items()):
        winners[min(indices, key=lambda i: (float(scores[i]), int(i)))] = indices
    selected = np.asarray(sorted(winners), dtype=np.int64)
    info = {
        i: {"group_size": len(indices), "candidate_source_indices": list(map(int, indices))}
        for i, indices in winners.items()
    }
    return selected, deficits, info


def _subset_face_targets(face_targets, indices):
    k = len(np.asarray(face_targets["face_rows"]))
    out = dict(face_targets)
    for key in ("face_rows", "face_nodes", "barycentric_weights", "face_samples_m",
                "face_targets_f32_m", "face_directions"):
        value = np.asarray(face_targets[key])
        if value.shape[0] != k:
            raise ValueError(f"{key} has unexpected row count")
        out[key] = value[indices].copy()
    archive = dict(face_targets.get("face_npz", {}))
    for key, value in list(archive.items()):
        a = np.asarray(value)
        if a.ndim >= 1 and a.shape[0] == k:
            archive[key] = a[indices].copy()
    out["face_npz"] = archive
    return out


def _empty_point_targets(point_targets):
    out = dict(point_targets)
    out["point_rows"] = np.empty(0, dtype=np.int64)
    out["point_targets_f32_m"] = np.empty((0, 3), dtype=np.float32)
    out["point_directions"] = np.empty((0, 3), dtype=np.float64)
    return out


def build_face_bone_grouped_clearance_rows(points_mm, faces, support, anchors,
                                             point_targets, face_targets, bary_owner,
                                             original_builder):
    """Compose exact frozen-408 point rows with one selected plane per group."""
    full_C, full_h, full_w, full_meta = original_builder(
        points_mm, faces, support, anchors, point_targets, face_targets, bary_owner)
    npoint = len(np.asarray(point_targets["point_rows"], dtype=np.int64))
    k = len(np.asarray(face_targets["face_rows"], dtype=np.int64))
    if (full_C.shape[0] != npoint + k or len(full_h) != npoint + k
            or len(full_w) != npoint + k or len(full_meta) != npoint + k):
        raise ValueError("frozen 408 builder returned unexpected row counts")
    selected, _, group_info = select_one_plane_per_face_bone(points_mm, face_targets)
    selected_targets = _subset_face_targets(face_targets, selected)
    core429 = _load_429()
    face_C, face_h, face_w, face_meta = core429.build_whole_face_clearance_rows(
        points_mm, faces, support, anchors, _empty_point_targets(point_targets),
        selected_targets, bary_owner, original_builder)

    # Preserve the complete frozen 408 point block exactly.
    point_C = sparse.csr_matrix(full_C[:npoint], dtype=np.float64)
    point_h = np.asarray(full_h[:npoint], dtype=np.float64).copy()
    point_w = np.asarray(full_w[:npoint], dtype=np.float64).copy()
    point_meta = [dict(x) for x in full_meta[:npoint]]
    face_C = sparse.csr_matrix(face_C, dtype=np.float64)
    face_h = np.asarray(face_h, dtype=np.float64).copy()
    face_w = np.asarray(face_w, dtype=np.float64).copy()
    face_mass = 0.5 if npoint else 1.0
    if len(face_w):
        face_w *= face_mass
    for record, weight in zip(face_meta, face_w):
        source_index = int(selected[int(record["source_index"])])
        record["source_index"] = source_index
        record["selected_source_index"] = source_index
        record["selection_criterion"] = "minimum_maximum_positive_corner_deficit"
        record.update(group_info[source_index])
        record["normalized_weight"] = float(weight)
    C = sparse.vstack((point_C, face_C), format="csr")
    h = np.concatenate((point_h, face_h))
    weights = np.concatenate((point_w, face_w))
    return C, h, weights, point_meta + face_meta


def replacement_for(original_builder):
    """Capture frozen 408 and return an opt-in 435 builder replacement."""
    def replacement(points_mm, faces, support, anchors, point_targets, face_targets, bary_owner):
        return build_face_bone_grouped_clearance_rows(
            points_mm, faces, support, anchors, point_targets, face_targets,
            bary_owner, original_builder)
    replacement.__name__ = "build_clearance_rows_face_bone_grouped_435"
    return replacement
