"""Opt-in containment-first soft objective; exact surface gates remain mandatory."""
from functools import wraps
import numpy as np

def replacement_for(original):
    @wraps(original)
    def build(points_mm,faces,support,anchors,point_targets,face_targets,bary_owner):
        C,h,w,meta=original(points_mm,faces,support,anchors,point_targets,face_targets,bary_owner)
        n=len(point_targets["point_rows"])
        if n==0:raise ValueError("containment-first requires at least one point target")
        if len(meta)<n or any(x["kind"]!="point" for x in meta[:n]):raise ValueError("point block identity mismatch")
        weights=np.asarray(w[:n],float).copy()
        if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError("invalid point weights")
        weights/=weights.sum()
        records=[dict(x,normalized_weight=float(v),objective_variant="containment_first_443",surface_acceptance_deferred=False)for x,v in zip(meta[:n],weights)]
        return C[:n].copy(),np.asarray(h[:n]).copy(),weights,records
    return build
