"""Inclusive per-face AABB overlap row selector for local same-surface broad phases."""
import numpy as np

def overlapping_rows(lower: np.ndarray, upper: np.ndarray,
                     query_lower: np.ndarray, query_upper: np.ndarray) -> np.ndarray:
    lo=np.asarray(lower,dtype=np.float64);hi=np.asarray(upper,dtype=np.float64)
    qlo=np.asarray(query_lower,dtype=np.float64);qhi=np.asarray(query_upper,dtype=np.float64)
    if lo.ndim!=2 or lo.shape[1:]!=(3,) or hi.shape!=lo.shape:
        raise ValueError("per-face bounds must be matching N by 3 arrays")
    if qlo.shape!=(3,) or qhi.shape!=(3,):
        raise ValueError("query bounds must be length-3 vectors")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or not np.isfinite(qlo).all() or not np.isfinite(qhi).all():
        raise ValueError("AABB bounds must be finite")
    if np.any(lo>hi) or np.any(qlo>qhi):
        raise ValueError("AABB lower bound exceeds upper bound")
    # Keep the reduction per row; inclusive comparisons retain boundary contacts.
    overlap=np.all(hi>=qlo[None,:],axis=1)&np.all(lo<=qhi[None,:],axis=1)
    return np.flatnonzero(overlap)
