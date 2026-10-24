"""Condition construction levels against edge-cut slivers; not an acceptance gate."""
import numpy as np


def select_conditioned_level(values, edges, target, maximum_shift):
    """Maximize the minimum interpolation fraction within a declared window.

    The crossing-edge set is constant between consecutive source-node values.
    Within each interval a bounded binary solve maximizes the minimum distance
    from either edge endpoint, measured as a fraction of that edge. Every
    interval in the window is considered; ties prefer the original target.
    This changes the inferred construction level, not source coordinates.
    Anatomical placement, volume, collision and cell-quality gates remain.
    """
    p=np.asarray(values,dtype=np.float64);e=np.asarray(edges)
    if p.ndim!=1 or not np.isfinite(p).all() or e.ndim!=2 or e.shape[1]!=2 or e.dtype.kind not in "iu" or np.any(e<0) or np.any(e>=len(p)):
        raise ValueError("finite nodal values and integer edge identities required")
    if not np.isfinite([target,maximum_shift]).all() or maximum_shift<=0:
        raise ValueError("finite positive construction window required")
    lo=float(target-maximum_shift);hi=float(target+maximum_shift)
    a=np.minimum(p[e[:,0]],p[e[:,1]]);b=np.maximum(p[e[:,0]],p[e[:,1]])
    events=np.unique(np.r_[lo,p[(p>lo)&(p<hi)],hi])
    best=None
    for left,right in zip(events[:-1],events[1:]):
        midpoint=left+(right-left)*.5
        active=(a<midpoint)&(b>midpoint)
        if not active.any():continue
        low=a[active];high=b[active];delta=high-low
        def feasible(t):
            lower=max(float(left),float(np.max(low+t*delta)))
            upper=min(float(right),float(np.min(high-t*delta)))
            return lower,upper
        low_t=0.;high_t=.5
        for _ in range(48):
            trial=(low_t+high_t)*.5;l,u=feasible(trial)
            if l<=u:low_t=trial
            else:high_t=trial
        lower,upper=feasible(low_t)
        level=float(np.clip(target,lower,upper))
        # Never silently choose a zero-length edge cut at a source event.
        if not left<level<right:continue
        score=float(np.min(np.minimum((level-low)/delta,(high-level)/delta)))
        key=(score,-abs(level-target),-level)
        if best is None or key>best[0]:
            best=(key,level,int(active.sum()),(float(left),float(right)))
    if best is None:raise ValueError("no nonempty conditioned cut in construction window")
    return {"selected_level":best[1],"target_level":float(target),
            "maximum_shift":float(maximum_shift),"actual_shift":best[1]-float(target),
            "minimum_edge_fraction":best[0][0],"crossing_edges":best[2],
            "source_value_interval":best[3],"physical_admission":False}

