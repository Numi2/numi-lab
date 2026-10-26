"""Opt-in output-space correction columns for offline connected-surface construction.
This extends the existing parent basis; it does not change any physical solver.
"""
import numpy as np
from scipy import sparse

def enrich_point_basis(B, scope, anchors, target_rows):
    B = sparse.csr_matrix(B, dtype=np.float64)
    scope = np.asarray(scope)
    anchors = np.asarray(anchors)
    target_rows = np.asarray(target_rows)
    for name, arr in [("scope", scope), ("anchors", anchors), ("target_rows", target_rows)]:
        if arr.ndim != 1 or arr.dtype.kind not in "iu":
            raise ValueError(name + " must be one-dimensional integer rows")
        if np.any(arr < 0):
            raise ValueError(name + " contains negative rows")
    if B.shape[0] != len(scope) or not np.isfinite(B.data).all():
        raise ValueError("basis shape/data invalid")
    if len(np.unique(scope)) != len(scope) or np.any(np.diff(scope) <= 0):
        raise ValueError("scope must be strictly sorted")
    target_rows = np.unique(target_rows.astype(np.int64))
    if np.setdiff1d(target_rows, scope).size:
        raise ValueError("point targets escape the existing constrained scope")
    selected = np.setdiff1d(target_rows, anchors).astype(np.int64)
    loc = np.searchsorted(scope, selected)
    # Do not duplicate a preexisting isolated exact identity column.
    counts = np.diff(B.tocsc().indptr)
    skip = []
    for row, out in zip(loc, selected):
        lo, hi = B.indptr[row:row+2]
        if hi-lo == 1 and B.data[lo] == 1.0 and counts[B.indices[lo]] == 1:
            skip.append(int(out))
    selected = np.setdiff1d(selected, np.asarray(skip, dtype=np.int64))
    loc = np.searchsorted(scope, selected)
    E = sparse.csr_matrix((np.ones(len(selected)), (loc, np.arange(len(selected)))),
                          shape=(len(scope), len(selected)))
    enriched = sparse.hstack((B, E), format="csr") if len(selected) else B.copy()
    anchor_mask = np.isin(scope, anchors)
    if enriched[anchor_mask].nnz:
        raise ValueError("enrichment or input basis moves a typed fixed anchor")
    report = dict(enabled=True, parent_column_count=B.shape[1], fine_column_count=len(selected),
                  requested_unique_point_rows=len(target_rows),
                  excluded_anchor_rows=np.intersect1d(target_rows, anchors).tolist(),
                  excluded_isolated_identity_rows=skip,
                  fine_output_node_rows=selected.tolist(),
                  matrix_shape=list(enriched.shape), matrix_nnz=enriched.nnz,
                  semantics="d = [Q_selected, E_contained_point_rows] q; every original output-space hard constraint, trust/source bound and L1 residual remains")
    return enriched, report
