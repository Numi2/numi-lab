"""Fail-closed source identity gate for repaired physical outer-surface regions.

Parent-surface IDs and detailed-source face rows are separate typed domains.
Retained coarse faces may keep parent identities; every face declared as part of a
source-guided replacement must carry a valid detailed-source face identity and
all its corners must have a compatible source embedding.
"""
from __future__ import annotations

import numpy as np


def _arr(value, dtype=None):
    return np.asarray(value, dtype=dtype)


def _identity_array(value, name, shape):
    """Validate integer identity data before any cast can truncate it."""
    if isinstance(value, (list, tuple)) and len(value) == 0 and shape == (None,):
        return np.empty((0,), dtype=np.int64), None
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        return None, f"{name}_dtype_not_integer"
    if raw.ndim != len(shape) or any(want is not None and got != want
                                     for got, want in zip(raw.shape, shape)):
        return None, f"{name}_shape_invalid"
    if raw.dtype.kind == "u" and raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        return None, f"{name}_value_out_of_int64_range"
    return raw.astype(np.int64, copy=False), None


def _face_domains(parent_ids, detailed_ids, source_face_count):
    p = np.asarray(parent_ids, dtype=np.int64)
    d = np.asarray(detailed_ids, dtype=np.int64)
    out = np.full(len(p), "unmapped", dtype="U12")
    pv = p >= 0
    dv = (d >= 0) & (d < source_face_count)
    out[pv & ~dv] = "parent"
    out[~pv & dv] = "detailed"
    out[pv & dv] = "mixed"
    out[(d >= source_face_count)] = "invalid"
    return out


def audit_registered_outer_source(
    outer_points,
    surface_faces,
    source_points,
    detailed_source_faces,
    *,
    parent_surface_face_ids,
    detailed_source_face_ids,
    required_replacement_face_rows=(),
    required_boundary_nodes=(),
    node_source_vertex_ids=None,
    node_source_face_rows=None,
    node_source_barycentric=None,
    node_source_edge_vertices=None,
    node_source_edge_weights=None,
    node_source_residual_world_m=None,
    point_tolerance_m=1e-7,
):
    """Audit exact typed identities and source embedding coverage for a physical O mesh.

    Parent-only faces are acceptable outside `required_replacement_face_rows`.
    Replacement faces must have detailed-face identity, and their three corners
    must be represented on that same detailed source triangle by a direct source
    vertex, a face+barycentric embedding, or an edge+barycentric embedding.
    Direct identities cannot carry a nonzero projection residual; projected rows
    must use face/edge embedding plus residual explicitly.
    """
    try:
        raw_tolerance = np.asarray(point_tolerance_m)
    except (TypeError, ValueError, OverflowError):
        return {"passed": False, "issues": ["point_tolerance_not_finite_positive_scalar"]}
    if raw_tolerance.ndim != 0 or raw_tolerance.dtype.kind not in "iuf":
        return {"passed": False, "issues": ["point_tolerance_not_finite_positive_scalar"]}
    tolerance = float(raw_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        return {"passed": False, "issues": ["point_tolerance_not_finite_positive_scalar"]}

    X = _arr(outer_points, np.float64)
    XS = _arr(source_points, np.float64)
    issues = []
    if X.ndim != 2 or X.shape[1:] != (3,) or not np.isfinite(X).all():
        return {"passed": False, "issues": ["outer_points_shape_or_finiteness"]}
    if XS.ndim != 2 or XS.shape[1:] != (3,) or not np.isfinite(XS).all():
        return {"passed": False, "issues": ["source_points_shape_or_finiteness"]}
    n, ns = len(X), len(XS)

    F, error = _identity_array(surface_faces, "surface_faces", (None, 3))
    if error:
        return {"passed": False, "issues": [error]}
    if not len(F) or np.any(F < 0) or np.any(F >= n):
        return {"passed": False, "issues": ["surface_faces_shape_or_index"]}
    FS, error = _identity_array(detailed_source_faces, "detailed_source_faces", (None, 3))
    if error:
        return {"passed": False, "issues": [error]}
    if not len(FS) or np.any(FS < 0) or np.any(FS >= ns):
        return {"passed": False, "issues": ["detailed_source_faces_shape_or_index"]}

    pids, error = _identity_array(parent_surface_face_ids, "parent_surface_face_ids", (len(F),))
    if error:
        return {"passed": False, "issues": [error]}
    dids, error = _identity_array(detailed_source_face_ids, "detailed_source_face_ids", (len(F),))
    if error:
        return {"passed": False, "issues": [error]}
    domains = _face_domains(pids, dids, len(FS))
    if np.any(domains == "invalid"):
        issues.append("detailed_source_face_identity_out_of_range")

    direct = np.full(n, -1, np.int64) if node_source_vertex_ids is None else node_source_vertex_ids
    face_row = np.full(n, -1, np.int64) if node_source_face_rows is None else node_source_face_rows
    bary = np.full((n, 3), np.nan) if node_source_barycentric is None else _arr(node_source_barycentric, np.float64)
    edge = np.full((n, 2), -1, np.int64) if node_source_edge_vertices is None else node_source_edge_vertices
    edge_w = np.full((n, 2), np.nan) if node_source_edge_weights is None else _arr(node_source_edge_weights, np.float64)
    residual = np.zeros((n, 3), np.float64) if node_source_residual_world_m is None else _arr(node_source_residual_world_m, np.float64)
    if node_source_vertex_ids is not None:
        direct, error = _identity_array(direct, "node_source_vertex_ids", (n,))
        if error:
            return {"passed": False, "issues": [error]}
    if node_source_face_rows is not None:
        face_row, error = _identity_array(face_row, "node_source_face_rows", (n,))
        if error:
            return {"passed": False, "issues": [error]}
    if node_source_edge_vertices is not None:
        edge, error = _identity_array(edge, "node_source_edge_vertices", (n, 2))
        if error:
            return {"passed": False, "issues": [error]}
    if bary.shape != (n, 3) or edge_w.shape != (n, 2) or residual.shape != (n, 3):
        return {"passed": False, "issues": ["node_source_embedding_arrays_do_not_cover_outer_nodes"]}

    direct_valid = (direct >= 0) & (direct < ns)
    face_valid = (face_row >= 0) & (face_row < len(FS))
    edge_valid = (edge[:, 0] >= 0) & (edge[:, 1] >= 0) & (edge < ns).all(axis=1)
    face_bary_valid = np.isfinite(bary).all(axis=1) & (bary >= -1e-10).all(axis=1) & (bary <= 1.0 + 1e-10).all(axis=1) & (np.abs(bary.sum(axis=1) - 1.0) <= 1e-8)
    edge_w_valid = np.isfinite(edge_w).all(axis=1) & (edge_w >= -1e-10).all(axis=1) & (edge_w <= 1.0 + 1e-10).all(axis=1) & (np.abs(edge_w.sum(axis=1) - 1.0) <= 1e-8)
    face_map = face_valid & face_bary_valid
    edge_map = edge_valid & edge_w_valid
    map_count = direct_valid.astype(np.int8) + face_map.astype(np.int8) + edge_map.astype(np.int8)
    ambiguous = map_count > 1
    if np.any(ambiguous):
        issues.append("node_has_multiple_conflicting_source_identity_forms")

    residual_finite = np.isfinite(residual).all(axis=1)
    direct_residual = np.linalg.norm(residual, axis=1)
    direct_with_residual = direct_valid & np.isfinite(direct_residual) & (direct_residual > tolerance)
    if np.any(direct_with_residual & ~face_map & ~edge_map):
        issues.append("direct_source_identity_has_projection_residual_without_projected_embedding")

    reconstructed = np.full((n, 3), np.nan, np.float64)
    node_kind = np.full(n, "unmapped", dtype="U8")
    for i in np.flatnonzero(direct_valid & ~direct_with_residual):
        reconstructed[i] = XS[direct[i]] + residual[i]
        node_kind[i] = "direct"
    for i in np.flatnonzero(face_map):
        reconstructed[i] = bary[i] @ XS[FS[face_row[i]]] + residual[i]
        node_kind[i] = "face"
    for i in np.flatnonzero(edge_map & edge_w_valid):
        reconstructed[i] = edge_w[i] @ XS[edge[i]] + residual[i]
        node_kind[i] = "edge"
    mapped = node_kind != "unmapped"
    if np.any(mapped & ~residual_finite):
        issues.append("mapped_node_has_nonfinite_source_residual")
    recon_error = np.full(n, np.inf)
    recon_error[mapped] = np.linalg.norm(reconstructed[mapped] - X[mapped], axis=1)
    recon_error[~np.isfinite(recon_error)] = np.inf
    bad_recon = mapped & (recon_error > tolerance)
    if np.any(bad_recon):
        issues.append("source_point_plus_residual_does_not_reconstruct_physical_outer_node")

    repl, error = _identity_array(required_replacement_face_rows, "required_replacement_face_rows", (None,))
    if error:
        return {"passed": False, "issues": [error]}
    if np.any(repl < 0) or np.any(repl >= len(F)):
        issues.append("required_replacement_face_row_out_of_range")
        repl = repl[(repl >= 0) & (repl < len(F))]
    repl_detail_valid = (dids[repl] >= 0) & (dids[repl] < len(FS)) if len(repl) else np.empty(0, bool)
    if np.any(~repl_detail_valid):
        issues.append("replacement_face_lacks_typed_detailed_source_face_identity")
    corner_mismatch = []
    for fr in repl[repl_detail_valid]:
        source_face = int(dids[fr])
        src_tri = set(map(int, FS[source_face]))
        for node in F[fr]:
            node = int(node)
            if node_kind[node] == "direct":
                ok = int(direct[node]) in src_tri
            elif node_kind[node] == "face":
                ok = int(face_row[node]) == source_face
            elif node_kind[node] == "edge":
                ok = set(map(int, edge[node])).issubset(src_tri)
            else:
                ok = False
            if not ok:
                corner_mismatch.append((int(fr), node, source_face, str(node_kind[node])))
    if corner_mismatch:
        issues.append("replacement_face_corner_source_identity_mismatch")

    req_nodes, error = _identity_array(required_boundary_nodes, "required_boundary_nodes", (None,))
    if error:
        return {"passed": False, "issues": [error]}
    invalid_req_nodes = req_nodes[(req_nodes < 0) | (req_nodes >= n)]
    if len(invalid_req_nodes):
        issues.append("required_boundary_node_out_of_range")
    req_nodes = req_nodes[(req_nodes >= 0) & (req_nodes < n)]
    bad_boundary = req_nodes[~mapped[req_nodes] | bad_recon[req_nodes]]
    if len(bad_boundary):
        issues.append("required_physical_boundary_node_unmapped_or_not_reconstructed")

    boundary_edges = {}
    for fi, tri in enumerate(F):
        for a, b in zip(tri, np.roll(tri, -1)):
            e = tuple(sorted((int(a), int(b))))
            boundary_edges[e] = boundary_edges.get(e, 0) + 1
    open_edges = [e for e, c in boundary_edges.items() if c == 1]
    open_nodes = sorted({v for e in open_edges for v in e})
    return {
        "passed": not issues,
        "issues": sorted(set(issues)),
        "counts": {
            "outer_nodes": int(n), "outer_faces": int(len(F)),
            "face_domain_detailed": int(np.sum(domains == "detailed")),
            "face_domain_parent_only": int(np.sum(domains == "parent")),
            "face_domain_mixed": int(np.sum(domains == "mixed")),
            "face_domain_unmapped": int(np.sum(domains == "unmapped")),
            "face_domain_invalid": int(np.sum(domains == "invalid")),
            "replacement_faces_required": int(len(repl)),
            "replacement_faces_missing_detailed_identity": int(np.sum(~repl_detail_valid)),
            "replacement_face_corner_mismatches": int(len(corner_mismatch)),
            "nodes_direct": int(np.sum(node_kind == "direct")),
            "nodes_face_embedded": int(np.sum(node_kind == "face")),
            "nodes_edge_embedded": int(np.sum(node_kind == "edge")),
            "nodes_unmapped_or_invalid": int(np.sum(node_kind == "unmapped")),
            "direct_ids_with_unrepresented_residual": int(np.sum(direct_with_residual & ~face_map & ~edge_map)),
            "nodes_reconstruction_error_over_tolerance": int(np.sum(bad_recon)),
            "required_boundary_nodes": int(len(req_nodes)),
            "required_boundary_nodes_unmapped_or_bad": int(len(bad_boundary)),
            "open_surface_boundary_edges": int(len(open_edges)),
            "open_surface_boundary_nodes": int(len(open_nodes)),
        },
        "max_source_reconstruction_error_m": float(np.max(recon_error[mapped])) if np.any(mapped) else None,
        "max_direct_projection_residual_m": float(np.max(direct_residual[direct_valid & np.isfinite(direct_residual)])) if np.any(direct_valid & np.isfinite(direct_residual)) else None,
        "replacement_face_identity_witnesses": [
            {"surface_face_row": int(r), "role_face_identity": "exact row in this pinned surface table", "domain": str(domains[r]), "parent_surface_id": int(pids[r]), "detailed_source_face_id": int(dids[r])}
            for r in repl[:32]
        ],
        "replacement_corner_mismatch_witnesses": [
            {"surface_face_row": f, "outer_node_row": v, "detailed_source_face_id": s, "node_mapping_kind": k}
            for f, v, s, k in corner_mismatch[:32]
        ],
        "unmapped_boundary_node_rows": [int(v) for v in open_nodes if not mapped[v]][:64],
    }
