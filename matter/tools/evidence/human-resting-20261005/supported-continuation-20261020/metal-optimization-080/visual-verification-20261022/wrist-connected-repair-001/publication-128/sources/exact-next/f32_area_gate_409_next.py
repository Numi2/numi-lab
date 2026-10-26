"""Exact changed-face emitted-F32 area guards for fixed-coordinate flips."""
from __future__ import annotations
import numpy as np

def f32_area_floor(points_trial_m, points_base_m, points_037_m, faces, face_rows=None, *, floor=0.1):
    """Apply the existing area-gate ratios to all or selected faces.

    Ratios use emitted Float32 trial coordinates, current-base orientation,
    and the matched fixed-037 area magnitude. Coordinates are in metres;
    cross-product magnitudes are evaluated in millimetres squared.
    """
    trial=np.asarray(points_trial_m)
    base=np.asarray(points_base_m)
    ref=np.asarray(points_037_m)
    f=np.asarray(faces)
    if trial.dtype!=np.float32 or base.dtype!=np.float32:
        raise ValueError("trial and base points must be emitted Float32")
    if trial.ndim!=2 or trial.shape[1]!=3 or base.shape!=trial.shape:
        raise ValueError("trial/base points must be matching Nx3 arrays")
    if ref.dtype!=np.float64 or ref.shape!=trial.shape:
        raise ValueError("matched-037 points must be F64 Nx3")
    if not np.isfinite(trial).all() or not np.isfinite(base).all() or not np.isfinite(ref).all():
        raise ValueError("point arrays must be finite")
    if f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer):
        raise ValueError("faces must be integer Fx3")
    if f.size and (int(f.min())<0 or int(f.max())>=len(trial)):
        raise ValueError("face index out of range")
    if not np.isfinite(floor) or floor<0 or floor>1:
        raise ValueError("area floor must be finite in [0,1]")
    if face_rows is None:
        rows=np.arange(len(f),dtype=np.int64)
    else:
        raw=np.asarray(face_rows)
        if raw.ndim!=1 or not np.issubdtype(raw.dtype,np.integer):
            raise ValueError("face_rows must be a one-dimensional integer array")
        rows=raw.astype(np.int64,copy=False)
        if len(np.unique(rows))!=len(rows): raise ValueError("duplicate face row")
        if rows.size and (rows.min()<0 or rows.max()>=len(f)): raise ValueError("face row out of range")
    if rows.size==0: raise ValueError("at least one face row is required")
    xx=trial.astype(np.float64)*1000.0
    bb=base.astype(np.float64)*1000.0
    rr=ref*1000.0
    selected=f[rows].astype(np.int64,copy=False)
    c0=np.cross(bb[selected[:,1]]-bb[selected[:,0]],bb[selected[:,2]]-bb[selected[:,0]])
    c1=np.cross(xx[selected[:,1]]-xx[selected[:,0]],xx[selected[:,2]]-xx[selected[:,0]])
    cr=np.cross(rr[selected[:,1]]-rr[selected[:,0]],rr[selected[:,2]]-rr[selected[:,0]])
    a0=np.linalg.norm(c0,axis=1); ar=np.linalg.norm(cr,axis=1)
    if np.any(a0<=0) or np.any(ar<=0) or not np.isfinite(c1).all():
        raise ValueError("area gate encountered degenerate/nonfinite geometry")
    orientation=np.einsum("ij,ij->i",c1,c0)/(a0*a0)
    magnitude=np.linalg.norm(c1,axis=1)/ar
    tol=float(floor)-2e-6
    bad_o=np.flatnonzero(orientation<tol); bad_r=np.flatnonzero(magnitude<tol)
    return {
      "face_rows":rows.tolist(),
      "minimum_current_oriented_ratio":float(orientation.min(initial=np.inf)),
      "minimum_fixed_037_reference_magnitude_ratio":float(magnitude.min(initial=np.inf)),
      "bad_current_oriented_face_rows":rows[bad_o].tolist(),
      "bad_fixed_037_reference_face_rows":rows[bad_r].tolist(),
      "area_floor":float(floor),
      "pass":bool(len(bad_o)==0 and len(bad_r)==0),
      "scope":"selected faces only" if face_rows is not None else "all faces"
    }
