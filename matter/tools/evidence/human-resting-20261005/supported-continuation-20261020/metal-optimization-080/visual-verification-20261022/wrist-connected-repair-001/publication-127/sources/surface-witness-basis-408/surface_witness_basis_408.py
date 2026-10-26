"""Surface-witness enrichment for the existing offline constrained parent basis."""
import numpy as np
from fine_point_basis_405 import enrich_point_basis

def enrich_surface_basis(B, scope, anchors, point_rows, face_nodes):
    face_nodes = np.asarray(face_nodes)
    if face_nodes.ndim != 2 or face_nodes.shape[1] != 3 or face_nodes.dtype.kind not in "iu":
        raise ValueError("face witness nodes must be integer triangles")
    if np.any(face_nodes < 0):
        raise ValueError("face witness nodes must be nonnegative")
    point_rows = np.asarray(point_rows)
    if point_rows.ndim != 1 or point_rows.dtype.kind not in "iu" or np.any(point_rows < 0):
        raise ValueError("point witness rows must be nonnegative integer rows")
    surface_rows = np.unique(face_nodes.ravel()).astype(np.int64)
    requested = np.union1d(point_rows, surface_rows).astype(np.int64)
    enriched, report = enrich_point_basis(B, scope, anchors, requested)
    fine = np.asarray(report["fine_output_node_rows"], dtype=np.int64)
    origin = np.isin(fine, point_rows).astype(np.uint8) + 2*np.isin(fine, surface_rows).astype(np.uint8)
    report["fine_output_node_origin_flags_408"] = origin.tolist()
    report["fine_output_node_origin_semantics_408"] = "bit1=contained-point target; bit2=current surface-witness corner; 3=both"
    report.update(requested_unique_point_rows=len(np.unique(point_rows)),
                  requested_unique_surface_rows=len(surface_rows),
                  requested_unique_union_rows=len(requested),
                  selection="contained point nodes and all corners of current skin-bone surface witnesses",
                  semantics="d = [Q_selected, E_current_witness_rows] q; original output-space hard constraints, source/trust bounds and material ancestry unchanged")
    return enriched, report
