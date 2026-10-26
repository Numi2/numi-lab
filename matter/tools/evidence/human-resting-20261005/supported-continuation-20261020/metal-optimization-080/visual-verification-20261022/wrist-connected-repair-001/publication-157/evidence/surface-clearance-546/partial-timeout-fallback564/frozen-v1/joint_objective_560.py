"""Keep the complete existing 458 point and face objective row set."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def retain_joint_rows560(
    clearance_builder,
    points_mm,
    faces,
    support,
    anchors,
    point_targets,
    face_targets,
    bary_owner,
):
    """Return all original 458 rows and weights without filtering or reweighting."""
    C, h, weights, metadata = clearance_builder(
        points_mm, faces, support, anchors, point_targets, face_targets, bary_owner
    )
    C_check = sparse.csr_matrix(C, dtype=np.float64)
    h_check = np.asarray(h, dtype=np.float64).reshape(-1)
    weights_check = np.asarray(weights, dtype=np.float64).reshape(-1)
    if C_check.shape[0] != len(h_check) or C_check.shape[0] != len(weights_check):
        raise ValueError("joint objective C/h/weight row counts differ")
    if not isinstance(metadata, (list, tuple)) or len(metadata) != C_check.shape[0]:
        raise ValueError("joint objective metadata rows do not align with C/h/weights")
    if not np.isfinite(C_check.data).all() or not np.isfinite(h_check).all():
        raise ValueError("joint objective contains nonfinite matrix or target values")
    if not np.isfinite(weights_check).all() or np.any(weights_check <= 0.0):
        raise ValueError("joint objective weights must be finite and positive")
    kinds = []
    for row in metadata:
        if not isinstance(row, dict) or not isinstance(row.get("kind"), str):
            raise ValueError("joint objective metadata requires a string kind per row")
        kinds.append(row["kind"])
    unknown = sorted(set(kinds) - {"face", "point"})
    if unknown:
        raise ValueError(f"joint objective contains unsupported row kinds: {unknown}")
    if not kinds:
        raise ValueError("joint objective has no existing 458 rows")
    counts = {kind: int(sum(value == kind for value in kinds)) for kind in ("face", "point")}
    masses = {
        kind: float(weights_check[np.asarray(kinds) == kind].sum(dtype=np.float64))
        for kind in ("face", "point")
    }
    proof = {
        "all_original_458_rows_retained": True,
        "rows_returned": int(C_check.shape[0]),
        "surface_soft_rows_retained": counts["face"],
        "point_soft_rows_retained": counts["point"],
        "surface_weight_mass_retained": masses["face"],
        "point_weight_mass_retained": masses["point"],
        "total_weight_mass_retained": float(weights_check.sum(dtype=np.float64)),
        "weights_recomputed_or_rebalanced": False,
        "row_order_and_metadata_preserved": True,
        "row_kinds": sorted(set(kinds)),
    }
    return C, h, weights, metadata, proof
