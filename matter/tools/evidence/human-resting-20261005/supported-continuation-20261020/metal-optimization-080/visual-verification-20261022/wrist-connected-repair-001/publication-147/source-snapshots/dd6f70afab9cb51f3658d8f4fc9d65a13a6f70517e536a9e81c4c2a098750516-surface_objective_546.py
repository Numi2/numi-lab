"""Surface-only row selection for the frozen 545 full-coordinate fit path."""
from __future__ import annotations
import numpy as np
from scipy import sparse


def retain_surface_rows546(clearance_builder, points_mm, faces, support, anchors,
                           point_targets, face_targets, bary_owner):
    """Keep every 458 face-plane objective row; require a genuinely empty point goal."""
    point_rows = np.asarray(point_targets.get("point_rows", np.empty(0)), dtype=np.int64)
    if point_rows.ndim != 1 or len(point_rows):
        raise ValueError("surface-only fit requires an empty point-target objective")
    C, h, weights, metadata = clearance_builder(
        points_mm, faces, support, anchors, point_targets, face_targets, bary_owner)
    C = sparse.csr_matrix(C, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if C.shape[0] != len(h) or C.shape[0] != len(weights) or len(metadata) != C.shape[0]:
        raise ValueError("surface objective row arrays do not align")
    kinds = [str(row.get("kind", "")) for row in metadata]
    keep = np.asarray([kind == "face" for kind in kinds], dtype=bool)
    if len(keep) != len(kinds) or not np.all(keep):
        raise ValueError("surface-only builder returned non-face objective rows")
    if not len(keep):
        raise ValueError("surface-only fit has no positive face-plane objective rows")
    if (not np.isfinite(C.data).all() or not np.isfinite(h).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0.0)):
        raise ValueError("surface objective contains nonfinite rows or nonpositive weights")
    mass = float(weights.sum(dtype=np.float64))
    if not np.isclose(mass, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"surface-only target weights must preserve the 458 total mass of 1, got {mass}")
    rows = [dict(row) for row in metadata]
    return C, h.copy(), weights.copy(), rows, {
        "point_target_rows": 0,
        "surface_soft_rows_retained": int(len(rows)),
        "surface_weight_mass_preserved": mass,
        "surface_row_kinds": sorted(set(kinds)),
        "all_458_surface_soft_rows_retained": True,
        "reweighted_rows": False,
        "objective_scope": "all movable positive-corner 458 face-plane rows; per-pair normalized weights retained",
    }


def select_outside_depth_rows546(node_rows, normals, lower, typed_ids=None):
    """Retain the already-outside depth rows and drop inside monotonicity rows."""
    node_rows = np.asarray(node_rows, dtype=np.int64).reshape(-1)
    normals = np.asarray(normals, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    if normals.shape != (len(node_rows), 3) or lower.shape != (len(node_rows),):
        raise ValueError("depth row arrays have inconsistent shapes")
    if not np.isfinite(normals).all() or not np.isfinite(lower).all():
        raise ValueError("depth rows contain nonfinite values")
    keep = lower < 0.0
    if not np.any(keep):
        raise ValueError("surface-only fit requires at least one outside depth guard row")
    ids = None
    if typed_ids is not None:
        ids = np.asarray(typed_ids)
        if ids.ndim == 0 or len(ids) != len(node_rows):
            raise ValueError("depth typed IDs do not align with depth rows")
        ids = ids[keep].copy()
    return (node_rows[keep].copy(), normals[keep].copy(), lower[keep].copy(), ids,
            {"input_depth_rows": int(len(node_rows)),
             "retained_outside_depth_rows": int(np.count_nonzero(keep)),
             "omitted_inside_monotonicity_rows": int(np.count_nonzero(~keep)),
             "outside_row_sign_rule": "retain only lower < 0 from candidate-matched 357 rows"})
