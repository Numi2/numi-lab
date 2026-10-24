"""Fail-closed validator for explicit 018 child-node -> original-013 material ancestry.

Diagnostic utility only: it does not construct ancestry or integrate any candidate.
"""
from __future__ import annotations
import numpy as np

class MappingError(ValueError):
    pass

def validate_reference_parent_map(*, node_parent_tet_rows, node_barycentric,
                                  parent_tetrahedra, deformed_parent_points, reference_parent_points,
                                  candidate_node_points, supplied_domain: str, expected_domain: str,
                                  expected_node_count: int,
                                  bary_tolerance: float = 1e-10,
                                  point_tolerance_m: float = 1e-8) -> dict:
    """Validate a local Lagrangian point map before composing any mass CSR.

    Parent IDs are typed by a caller-supplied domain string, so a numerically
    in-range merged/TetGen row cannot masquerade as an original 013 row.
    Inverse barycentric pullback is rejected when any referenced deformed parent
    tet has nonpositive orientation.
    """
    if supplied_domain != expected_domain:
        raise MappingError(f"parent row domain mismatch: {supplied_domain!r} != {expected_domain!r}")
    if node_parent_tet_rows is None or node_barycentric is None:
        raise MappingError("explicit per-node parent tetrahedron and barycentric ancestry are required")
    rows = np.asarray(node_parent_tet_rows)
    bary = np.asarray(node_barycentric)
    pt = np.asarray(parent_tetrahedra)
    X = np.asarray(deformed_parent_points)
    R = np.asarray(reference_parent_points)
    C = np.asarray(candidate_node_points)
    if rows.dtype.kind not in "iu" or rows.shape != (expected_node_count,):
        raise MappingError("node parent rows must be an integer array aligned to every candidate node")
    if bary.shape != (expected_node_count, 4) or not np.isfinite(bary).all():
        raise MappingError("node barycentrics must be finite [candidate_node_count,4]")
    if pt.dtype.kind not in "iu" or pt.ndim != 2 or pt.shape[1] != 4 or len(pt) == 0:
        raise MappingError("typed parent tetrahedra must be [P,4]")
    if (X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all() or
            R.shape != X.shape or not np.isfinite(R).all() or
            C.shape != (expected_node_count, 3) or not np.isfinite(C).all() or
            pt.min() < 0 or pt.max() >= len(X)):
        raise MappingError("parent coordinates, candidate points, or tetrahedron node indices are invalid")
    if rows.min(initial=0) < 0 or rows.max(initial=-1) >= len(pt):
        raise MappingError("candidate node ancestry references an out-of-domain parent row")
    if not np.isfinite(bary_tolerance) or bary_tolerance < 0:
        raise MappingError("barycentric tolerance must be finite and nonnegative")
    if not np.isfinite(point_tolerance_m) or point_tolerance_m <= 0:
        raise MappingError("point tolerance must be finite and positive")
    sums = bary.sum(axis=1)
    if np.any(np.abs(sums - 1.0) > bary_tolerance) or np.any(bary < -bary_tolerance):
        raise MappingError("candidate node barycentrics do not form positive unit-sum coordinates")
    mapped = pt[rows]
    q = X[mapped].astype(np.float64)
    reconstructed = np.einsum("ni,nij->nj", bary, q)
    residual = np.linalg.norm(reconstructed - C.astype(np.float64), axis=1)
    if np.any(residual > point_tolerance_m):
        bad = int(np.argmax(residual))
        raise MappingError(f"parent row identity does not reconstruct candidate node {bad}; residual_m={residual[bad]:.9g}")
    reference_points = np.einsum("ni,nij->nj", bary, R[mapped].astype(np.float64))
    det = np.einsum("ij,ij->i", q[:,1]-q[:,0], np.cross(q[:,2]-q[:,0], q[:,3]-q[:,0]))
    if np.any(~np.isfinite(det)) or np.any(det <= 0):
        raise MappingError("inverse material pullback crosses a nonpositive deformed parent tet")
    return {"validated_nodes": int(expected_node_count), "min_parent_signed_det_m3x6": float(det.min()),
            "max_candidate_reconstruction_residual_m": float(residual.max(initial=0.0)),
            "mapped_reference_points": reference_points, "parent_domain": supplied_domain,
            "scope": "local cell validation only; not a global bijection certificate"}
