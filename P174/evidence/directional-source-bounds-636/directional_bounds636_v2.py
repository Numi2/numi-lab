"""Construction-only asymmetric boxes inside the existing trust/source balls.

For offset u=x-source, retain 408's old outward bound b and increase only
its inward bound in each component to min(trust/sqrt(3), 2*abs(u_i)+b).
The box contains 408's old cube and zero. Every corner satisfies both balls:
max |d_i| <= trust/sqrt(3), max |u_i+d_i| <= |u_i|+b.
"""
from __future__ import annotations
import numpy as np

def directional_bounds(points_mm, source_mm, support, anchors, trust_mm, source_cap_mm):
    p=np.asarray(points_mm,dtype=np.float64);src=np.asarray(source_mm,dtype=np.float64)
    support=np.asarray(support,dtype=np.int64);anchors=np.asarray(anchors,dtype=np.int64)
    trust=float(trust_mm);cap=float(source_cap_mm)
    if p.ndim!=2 or p.shape[1]!=3 or src.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(src).all():
        raise ValueError('point/source must be matching finite (N,3) arrays')
    if support.ndim!=1 or len(np.unique(support))!=len(support) or np.any(support<0) or np.any(support>=len(p)):
        raise ValueError('invalid support')
    if anchors.ndim!=1 or np.any(anchors<0) or np.any(anchors>=len(p)):
        raise ValueError('invalid anchors')
    if not np.isfinite([trust,cap]).all() or trust<=0 or cap<=0:
        raise ValueError('invalid trust/source cap')
    all_distance=np.linalg.norm(p-src,axis=1)
    if np.any(all_distance>cap):raise ValueError('starting source point exceeds cap')
    offset=p[support]-src[support];distance=all_distance[support]
    b=np.minimum(trust,np.maximum(0.0,cap-distance))/np.sqrt(3.)
    inward=np.minimum(trust/np.sqrt(3.),2*np.abs(offset)+b[:,None])
    lo=np.where(offset>0,-inward,-b[:,None]);hi=np.where(offset<0,inward,b[:,None])
    fixed=np.isin(support,anchors);lo[fixed]=0.;hi[fixed]=0.
    if np.any(lo>0) or np.any(hi<0) or np.any(lo>hi):raise AssertionError('baseline excluded')
    worst_step=np.linalg.norm(np.maximum(np.abs(lo),np.abs(hi)),axis=1)
    worst_source=np.linalg.norm(np.maximum(np.abs(offset+lo),np.abs(offset+hi)),axis=1)
    if np.any(worst_step>trust+1e-11) or np.any(worst_source>cap+1e-11):raise AssertionError('box escapes stated sphere')
    expanded=(np.abs(lo)>b[:,None]+1e-12)|(np.abs(hi)>b[:,None]+1e-12)
    receipt={'semantics':'asymmetric inner box preserving zero, original 408 cube, and actual trust/source spheres',
      'trust_mm':trust,'source_cap_mm':cap,'support_nodes':len(support),'anchors_on_support':int(fixed.sum()),
      'expanded_component_count':int(expanded.sum()),'expanded_node_count':int(np.any(expanded,axis=1).sum()),
      'maximum_box_step_mm':float(worst_step.max(initial=0)),
      'maximum_box_source_distance_mm':float(worst_source.max(initial=0)),
      'native_admission':False,'accepted_physical_steps':0}
    return lo.ravel(),hi.ravel(),receipt
