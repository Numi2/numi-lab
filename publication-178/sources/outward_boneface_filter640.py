"""Narrow optional pruning of the anti-outward bone-face SAT sign.

The SAT inputs must preserve consistently outward-wound fixed-bone triangles.
Only the `bone_face` axis whose normal opposes that ordered triangle normal is
removed. Skin-face and edge-cross axes are never filtered.
"""
from __future__ import annotations
from typing import Any
import numpy as np

DOT_TOL = 1e-10

def filter_axis_inventory(axes: list[dict[str, Any]], bone_triangle_mm: np.ndarray, *, enabled: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tri=np.asarray(bone_triangle_mm,dtype=np.float64)
    if tri.shape!=(3,3) or not np.isfinite(tri).all():
        raise ValueError("fixed bone triangle must be finite shape (3,3)")
    edges=np.asarray([tri[1]-tri[0],tri[2]-tri[1],tri[0]-tri[2]])
    scale=max(float(np.max(np.linalg.norm(edges,axis=1))),1.0)
    raw=np.cross(tri[1]-tri[0],tri[2]-tri[0])
    norm=float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm<=1e-14*scale*scale:
        raise ValueError("fixed bone triangle is degenerate; outward orientation is undefined")
    outward=raw/norm
    kept=[]; removed=[]; bone_face_count=0; anti_count=0
    for index,axis in enumerate(axes):
        if not isinstance(axis,dict): raise ValueError("axis record must be a mapping")
        source=str(axis.get("axis_source",""))
        if source!="bone_face":
            kept.append(axis)
            continue
        bone_face_count+=1
        if int(axis.get("axis_index",-1))!=0:
            raise ValueError("bone_face SAT axis must have index zero")
        n=np.asarray(axis.get("normal_world_unit"),dtype=np.float64)
        if n.shape!=(3,) or not np.isfinite(n).all() or abs(float(np.linalg.norm(n))-1.)>2e-8:
            raise ValueError("bone_face SAT axis must be a finite unit normal")
        dot=float(np.dot(n,outward))
        # A bone-face axis is derived from this triangle's own winding. A
        # materially nonparallel value signals a mismatched triangle/axis pair.
        if abs(abs(dot)-1.)>DOT_TOL:
            raise ValueError("bone_face SAT axis does not match fixed triangle winding")
        if enabled and dot<=-1.+DOT_TOL:
            anti_count+=1
            removed.append({"input_index":index,"axis_source":source,"axis_index":0,
                            "axis_sign":int(axis.get("axis_sign",0)),
                            "extra_sign":int(axis.get("extra_sign",0)),
                            "outward_dot":dot})
        else:
            kept.append(axis)
    receipt={"schema":"numi.outward_boneface_filter640.receipt.v1",
        "enabled":bool(enabled),"rule":"remove only bone_face SAT axes with dot(axis, ordered fixed-bone triangle normal) <= -1 + 1e-10",
        "dot_tolerance":DOT_TOL,"input_axis_count":len(axes),"output_axis_count":len(kept),
        "bone_face_axis_count":bone_face_count,"opposed_bone_face_axis_count":anti_count,
        "removed_axis_count":len(removed),"removed_axes":removed,
        "non_bone_face_axes_preserved":True,"triangle_winding_is_assumed_outward":True}
    return kept,receipt
