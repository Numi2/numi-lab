"""Explicit failure-triggered scale policy for connected construction backtracking."""
from __future__ import annotations
import numpy as np

FAILURE_KINDS = {"area", "self", "point", "surface"}

def update_failure_scales(*, faces, scales, bad_faces, self_pairs, anchors,
                          fixed_prefix_nodes, zero_below, failure_kind,
                          uniform_failure_backtracking, local_reducer,
                          global_reducer, point_witness_faces=None):
    """Apply configured construction-only damping after an exact witness."""
    F = np.asarray(faces)
    old = np.asarray(scales, dtype=np.float64)
    bad = np.asarray(bad_faces, dtype=np.int64).reshape(-1)
    pairs = np.asarray(self_pairs, dtype=np.int64).reshape(-1, 2)
    anchors = np.asarray(anchors, dtype=np.int64).reshape(-1)
    n = len(old)
    point_witness = np.asarray([] if point_witness_faces is None else point_witness_faces,
                               dtype=np.int64).reshape(-1)
    if failure_kind not in FAILURE_KINDS:
        raise ValueError("unknown exact construction failure kind")
    if old.ndim != 1 or not np.isfinite(old).all() or np.any(old < 0) or np.any(old > 1):
        raise ValueError("invalid input construction scales")
    if F.ndim != 2 or F.shape[1] != 3 or (F.size and (np.any(F < 0) or np.any(F >= n))):
        raise ValueError("invalid face topology")
    if np.any(anchors < 0) or np.any(anchors >= n) or np.unique(anchors).size != len(anchors):
        raise ValueError("invalid anchor rows")
    if type(fixed_prefix_nodes) is not int or not 0 <= fixed_prefix_nodes <= n:
        raise ValueError("invalid fixed prefix")
    if not 0 < float(zero_below) < 1:
        raise ValueError("invalid construction cutoff")
    if len(point_witness) and (np.any(point_witness < 0) or np.any(point_witness >= len(F))):
        raise ValueError("point-witness face row out of range")
    if failure_kind == "point" and len(point_witness) == 0:
        raise ValueError("point regression requires its actual incident-face witnesses")
    if len(bad) == 0 and len(pairs) == 0 and len(point_witness) == 0:
        raise ValueError("failure-triggered update needs exact witnesses")
    if len(bad) and (np.any(bad < 0) or np.any(bad >= len(F))):
        raise ValueError("bad-face witness out of range")
    if len(pairs) and (np.any(pairs < 0) or np.any(pairs >= len(F)) or np.any(pairs[:, 0] == pairs[:, 1])):
        raise ValueError("self-pair witness out of range or degenerate")

    combined_bad = np.unique(np.concatenate((bad, point_witness)))
    if uniform_failure_backtracking:
        new, receipt = global_reducer(F, old, combined_bad, pairs, anchors=anchors,
                                      zero_below=float(zero_below))
        new = np.asarray(new, dtype=np.float64).copy()
        mode = "UNIFORM_FAILURE_BACKTRACKING"
    else:
        new, receipt = local_reducer(F, old, combined_bad, pairs, anchors=anchors,
                                     zero_below=float(zero_below))
        new = np.asarray(new, dtype=np.float64).copy()
        mode = "LOCAL_WITNESS_BACKTRACKING"
    if new.shape != old.shape or not np.isfinite(new).all() or np.any(new < 0) or np.any(new > old):
        raise ValueError("failure scale update is malformed or increases motion")
    new[anchors] = 0.0
    new[:fixed_prefix_nodes] = 0.0
    unchanged = np.array_equal(new, old)
    result = dict(receipt)
    result.update({"policy": mode, "failure_kind": failure_kind,
                   "failure_witness_face_rows": combined_bad.tolist(),
                   "failure_witness_self_pairs": pairs.tolist(),
                   "fixed_prefix_zeroed": int(fixed_prefix_nodes),
                   "anchors_zeroed": int(len(anchors)),
                   "unchanged_scales": bool(unchanged),
                   "construction_only": True,
                   "geometry_certified": False,
                   "native_admission": False,
                   "accepted_physical_steps": 0})
    if unchanged:
        result["status"] = "NO_CONSTRUCTION_SCALE_PROGRESS"
    return new, result
