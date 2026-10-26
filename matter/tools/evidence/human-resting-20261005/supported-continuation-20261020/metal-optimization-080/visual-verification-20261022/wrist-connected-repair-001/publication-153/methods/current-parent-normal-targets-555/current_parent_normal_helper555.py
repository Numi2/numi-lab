"""Numerical current-parent normal construction for refinement-only target proposals.

This module computes oriented area-weighted normals from immediate parent faces.
It does not certify clearance or change any mesh coordinates.
"""
from __future__ import annotations
import numpy as np


def _unit(v: np.ndarray, label: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{label} must be a finite 3-vector")
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n <= 1e-14:
        raise ValueError(f"{label} is degenerate")
    return a / n


def _validate_parent_arrays(parent_points, parent_faces, child_points,
                            node_kind, parent_face_indptr,
                            parent_face_rows, parent_face_barycentric):
    pp = np.asarray(parent_points)
    pf = np.asarray(parent_faces)
    cp = np.asarray(child_points)
    kinds = np.asarray(node_kind)
    ptr = np.asarray(parent_face_indptr)
    rows = np.asarray(parent_face_rows)
    bary = np.asarray(parent_face_barycentric)
    if pp.dtype != np.float32 or pp.ndim != 2 or pp.shape[1:] != (3,) or not np.isfinite(pp).all():
        raise ValueError("parent points must be finite Float32 Nx3")
    if cp.dtype != np.float32 or cp.ndim != 2 or cp.shape[1:] != (3,) or not np.isfinite(cp).all():
        raise ValueError("child points must be finite Float32 Nx3")
    if pf.ndim != 2 or pf.shape[1:] != (3,) or pf.dtype.kind not in "iu":
        raise ValueError("parent faces must be integer Fx3")
    if np.any(pf < 0) or np.any(pf >= len(pp)):
        raise ValueError("parent face node index is outside parent point array")
    if kinds.shape != (len(cp),) or kinds.dtype.kind not in "iu":
        raise ValueError("refinement node kinds are not aligned to child points")
    if ptr.shape != (len(cp) + 1,) or ptr.dtype.kind not in "iu":
        raise ValueError("parent-face CSR indptr is not aligned to child points")
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or bary.shape != (len(rows), 3):
        raise ValueError("parent-face CSR rows/barycentrics are malformed")
    if ptr[0] != 0 or ptr[-1] != len(rows) or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError("parent-face CSR pointers are malformed")
    if rows.size and (np.any(rows < 0) or np.any(rows >= len(pf))):
        raise ValueError("parent-face CSR references a face outside the immediate parent")
    if not np.isfinite(bary).all():
        raise ValueError("parent-face barycentrics contain nonfinite values")
    return (pp.astype(np.float64), pf.astype(np.int64), cp.astype(np.float64),
            kinds.astype(np.int64), ptr.astype(np.int64), rows.astype(np.int64),
            bary.astype(np.float64))


def parent_normal_for_node(node_row: int, *, parent_points, parent_faces,
                           child_points, node_kind, parent_face_indptr,
                           parent_face_rows, parent_face_barycentric,
                           outward_sign: int,
                           point_tolerance_m: float = 2e-7) -> tuple[np.ndarray, dict]:
    """Return the orientation-signed, current-parent area-weighted normal.

    Kind 2 must have one centroid parent face. Kind 1 must have two incident
    parent faces; their area-weighted outward unit normals are summed. The
    parent-face barycentrics must reconstruct the emitted Float32 child point.
    """
    if outward_sign not in (-1, 1):
        raise ValueError("outward_sign must be pinned to +1 or -1")
    if not np.isfinite(point_tolerance_m) or point_tolerance_m <= 0:
        raise ValueError("point_tolerance_m must be positive and finite")
    pp, pf, cp, kinds, ptr, rows, bary = _validate_parent_arrays(
        parent_points, parent_faces, child_points, node_kind,
        parent_face_indptr, parent_face_rows, parent_face_barycentric)
    node = int(node_row)
    if node < 0 or node >= len(cp):
        raise ValueError("node row is outside child points")
    kind = int(kinds[node])
    if kind not in (1, 2):
        raise ValueError(f"node {node} is not an appended edge/centroid refinement node")
    lo, hi = int(ptr[node]), int(ptr[node + 1])
    expected_count = 2 if kind == 1 else 1
    if hi - lo != expected_count:
        raise ValueError(f"refinement kind {kind} requires {expected_count} immediate parent faces")
    face_rows = rows[lo:hi]
    if len(np.unique(face_rows)) != len(face_rows):
        raise ValueError("parent-face ancestry contains duplicate faces")
    bary_rows = bary[lo:hi]
    if np.any(bary_rows < -1e-12) or np.any(bary_rows > 1.0 + 1e-12):
        raise ValueError("parent-face barycentrics fall outside the source triangle")
    if not np.allclose(np.sum(bary_rows, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("parent-face barycentrics do not sum to one")
    if kind == 2 and not np.allclose(bary_rows[0], np.full(3, 1.0 / 3.0), rtol=0.0, atol=1e-12):
        raise ValueError("kind-2 node is not the immediate parent-face centroid")
    if kind == 1 and not np.allclose(np.sort(bary_rows, axis=1),
                                     np.tile([0.0, 0.5, 0.5], (2, 1)),
                                     rtol=0.0, atol=1e-12):
        raise ValueError("kind-1 node is not the immediate parent-edge midpoint")

    face_normals = []
    face_areas = []
    reconstructions = []
    for face_row, weights in zip(face_rows, bary_rows):
        tri = pp[pf[int(face_row)]]
        reconstructed = weights @ tri
        reconstructions.append(reconstructed)
        raw = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        area2 = float(np.linalg.norm(raw))
        if not np.isfinite(area2) or area2 <= 1e-14:
            raise ValueError(f"immediate parent face {int(face_row)} is degenerate")
        face_areas.append(0.5 * area2)
        face_normals.append((int(outward_sign) / area2) * raw)
    for face_row, reconstructed in zip(face_rows, reconstructions):
        if float(np.max(np.abs(reconstructed - cp[node]))) > point_tolerance_m:
            raise ValueError(f"node {node} does not match parent-face barycentric ancestry {int(face_row)}")
    weighted = np.sum(np.asarray(face_areas)[:, None] * np.asarray(face_normals), axis=0)
    normal = _unit(weighted, f"area-weighted parent normal for node {node}")
    if len(face_normals) == 2 and float(np.dot(face_normals[0], face_normals[1])) <= -0.999999:
        raise ValueError(f"edge midpoint {node} has opposing parent-face normals")
    provenance = {
        "method": "immediate-parent-face-area-weighted-oriented-normal",
        "node_kind": kind,
        "node_row": node,
        "parent_face_rows": [int(x) for x in face_rows],
        "parent_face_area_m2": [float(x) for x in face_areas],
        "parent_face_outward_normals_world_unit": [x.tolist() for x in face_normals],
        "parent_face_barycentric": bary_rows.tolist(),
        "orientation_sign_from_registered_coarse_source": int(outward_sign),
        "emitted_point_matches_parent_barycentrics_within_m": float(point_tolerance_m),
    }
    return normal, provenance


def parent_normal_field(target_rows, *, parent_points, parent_faces,
                        child_points, node_kind, parent_face_indptr,
                        parent_face_rows, parent_face_barycentric,
                        old_prefix_count: int, outward_sign: int,
                        point_tolerance_m: float = 2e-7) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
    rows = sorted(set(int(x) for x in target_rows))
    if any(row < int(old_prefix_count) for row in rows):
        raise ValueError("current-parent normal targets must be appended refinement nodes only")
    normals, provenance = {}, {}
    for row in rows:
        normals[row], provenance[row] = parent_normal_for_node(
            row, parent_points=parent_points, parent_faces=parent_faces,
            child_points=child_points, node_kind=node_kind,
            parent_face_indptr=parent_face_indptr,
            parent_face_rows=parent_face_rows,
            parent_face_barycentric=parent_face_barycentric,
            outward_sign=outward_sign, point_tolerance_m=point_tolerance_m)
    return normals, provenance

