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



def condition_nodal_cut_clearance(values, edges, levels, *, minimum_fraction,
                                  maximum_raise, fixed_nodes, iterations=16):
    """Monotonically condition an inferred field without moving source points.

    Raising P1 coefficients preserves every nonnegative-barycentric lower-bound
    constraint. Values may cross a material level; this changes construction
    geometry and must pass the unchanged mass, thickness and cell-quality gates.
    Fixed outer source values remain bit-exact. A bounded unsuccessful trial is
    explicitly returned with unresolved crossing edges; it is never admission.
    """
    original=np.asarray(values,dtype=float); e=np.asarray(edges); fixed=np.asarray(fixed_nodes)
    lev=np.asarray(levels,dtype=float)
    if (original.ndim!=1 or not np.isfinite(original).all() or e.ndim!=2 or e.shape[1]!=2 or
        e.dtype.kind not in "iu" or np.any(e<0) or np.any(e>=len(original)) or
        fixed.shape!=original.shape or fixed.dtype.kind!="b" or lev.ndim!=1 or not len(lev) or
        not np.isfinite(lev).all() or np.any(np.diff(lev)<=0) or
        not np.isfinite([minimum_fraction,maximum_raise]).all() or
        not 0<minimum_fraction<.5 or maximum_raise<=0 or
        not isinstance(iterations,int) or isinstance(iterations,bool) or iterations<1):
        raise ValueError("invalid bounded nodal conditioning domain")
    p=original.copy(); done=0; limit=original+maximum_raise
    for done in range(1,iterations+1):
        proposed=p.copy()
        for level in lev:
            va=p[e[:,0]];vb=p[e[:,1]]
            low=np.minimum(va,vb);high=np.maximum(va,vb)
            active=(low<level)&(high>level)
            rows=np.flatnonzero(active)
            if not len(rows):continue
            a=e[rows,0];b=e[rows,1]
            lower=np.where(va[rows]<=vb[rows],a,b)
            upper=np.where(va[rows]<=vb[rows],b,a)
            lo=low[rows];hi=high[rows]
            fraction=(level-lo)/(hi-lo)
            # Move a nearly cut lower endpoint across the level. A later pass
            # enforces clearance on any newly crossing adjacent edge.
            bad=fraction<minimum_fraction
            bump=np.nextafter(level+minimum_fraction/(1-minimum_fraction)*(hi[bad]-level),np.inf)
            np.maximum.at(proposed,lower[bad],bump)
            bad=(1-fraction)<minimum_fraction
            bump=np.nextafter((level-minimum_fraction*lo[bad])/(1-minimum_fraction),np.inf)
            np.maximum.at(proposed,upper[bad],bump)
        proposed=np.minimum(proposed,limit);proposed[fixed]=original[fixed]
        if np.array_equal(proposed,p):break
        p=proposed
    unresolved=0;minimum=.5;crossing=0
    for level in lev:
        a=np.minimum(p[e[:,0]],p[e[:,1]]);b=np.maximum(p[e[:,0]],p[e[:,1]])
        active=(a<level)&(b>level)
        if active.any():
            f=(level-a[active])/(b[active]-a[active]);q=np.minimum(f,1-f)
            minimum=min(minimum,float(q.min()));crossing+=len(q)
            unresolved+=int(np.count_nonzero(q<minimum_fraction))
    assert np.all(p>=original) and np.array_equal(p[fixed],original[fixed])
    return p,{"changed_nodes":int(np.count_nonzero(p!=original)),
        "maximum_actual_raise":float(np.max(p-original,initial=0)),
        "maximum_allowed_raise":float(maximum_raise),"minimum_fraction_requested":float(minimum_fraction),
        "minimum_crossing_edge_fraction":minimum,"crossing_edges":crossing,
        "unresolved_edges":unresolved,"iterations":done,
        "conditioning_complete":unresolved==0,"physical_admission":False}
