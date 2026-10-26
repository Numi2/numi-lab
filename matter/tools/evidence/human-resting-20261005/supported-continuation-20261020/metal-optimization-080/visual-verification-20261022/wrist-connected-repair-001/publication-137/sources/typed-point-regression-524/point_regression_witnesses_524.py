"""Select actual penetration regressions from complete typed before/after tables."""
import numpy as np

def witness_depths(summary, node_count):
    rows = summary.get("inside_witnesses")
    count = summary.get("inside_node_bank_pairs")
    if not isinstance(rows, list) or type(count) is not int or count != len(rows):
        raise ValueError("complete typed inside witness table required")
    result = {}
    for w in rows:
        if w.get("classification") != "inside":
            raise ValueError("non-inside witness in inside table")
        i = int(w["output_node"])
        if i != int(w["query_row"]) or not 0 <= i < node_count:
            raise ValueError("inconsistent or out-of-domain witness node")
        key = (i, int(w["bank_body"]), int(w["bank_stable_id"]))
        depth = float(w["nearest_distance_m"])
        if key in result or not np.isfinite(depth) or depth < 0:
            raise ValueError("duplicate or invalid typed penetration witness")
        result[key] = depth
    return result

def regressed_point_faces(before, after, faces, node_count):
    F = np.asarray(faces)
    if F.ndim != 2 or F.shape[1] != 3 or F.dtype.kind not in "iu" or np.any(F < 0) or np.any(F >= node_count):
        raise ValueError("invalid face domain")
    old = witness_depths(before, node_count)
    new = witness_depths(after, node_count)
    # New typed penetrations and increased depths require local damping.
    # Existing nonincreasing penetrations are not treated as new failures.
    rows = np.asarray(sorted({k[0] for k,d in new.items() if k not in old or d > old[k]}), dtype=np.int64)
    incident = np.flatnonzero(np.isin(F, rows).any(axis=1))
    referenced = np.unique(F[incident][np.isin(F[incident], rows)])
    if not np.array_equal(referenced, rows):
        raise ValueError("unreferenced penetration witness")
    return rows, incident
