"""Sparse face-interior exit constraints for connected skin construction.

These are proposal constraints, not collision certification or physical state.
Exact emitted-coordinate geometry checks remain mandatory.
"""
import numpy as np
from scipy import sparse

def build_barycentric_exit_system(points_mm, faces, face_rows, weights, normals, targets_mm, *, anchor_rows=()):
    p=np.asarray(points_mm,dtype=np.float64)
    f=np.asarray(faces)
    rows=np.asarray(face_rows)
    w=np.asarray(weights,dtype=np.float64)
    n=np.asarray(normals,dtype=np.float64)
    t=np.asarray(targets_mm,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all(): raise ValueError("invalid points")
    if f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in "iu" or np.any(f<0) or np.any(f>=len(p)): raise ValueError("invalid faces")
    if rows.ndim!=1 or rows.dtype.kind not in "iu" or np.any(rows<0) or np.any(rows>=len(f)): raise ValueError("invalid face rows")
    k=len(rows)
    if w.shape!=(k,3) or n.shape!=(k,3) or t.shape!=(k,3) or not all(np.isfinite(x).all() for x in (w,n,t)): raise ValueError("invalid sparse target arrays")
    if np.any(w<0) or np.any(w>1) or np.max(np.abs(w.sum(axis=1)-1),initial=0)>2e-12: raise ValueError("nonconvex barycentric weights")
    if np.max(np.abs(np.linalg.norm(n,axis=1)-1),initial=0)>2e-10: raise ValueError("nonunit exit normals")
    anchors=np.asarray(anchor_rows)
    if anchors.size and (anchors.ndim!=1 or anchors.dtype.kind not in "iu" or np.any(anchors<0) or np.any(anchors>=len(p))): raise ValueError("invalid anchors")
    tri=f[rows].astype(np.int64)
    B=sparse.coo_matrix((w.ravel(),(np.repeat(np.arange(k),3),tri.ravel())),shape=(k,len(p))).tocsr()
    sample=np.asarray(B@p)
    delta=t-sample
    lower=np.einsum("ij,ij->i",n,delta)
    if np.any(lower<=0): raise ValueError("exit plane is not ahead of sample")
    g=w[:,:,None]*n[:,None,:]
    g[np.isin(tri,anchors)]=0.
    if np.any(np.einsum("mij,mij->m",g,g)<=0): raise ValueError("exit row has no free support")
    return B,tri,g,lower,delta

def progressive_barycentric_lower(g,tri,displacement,targets,trust_radius_mm,fraction=.95):
    g=np.asarray(g,dtype=np.float64); tri=np.asarray(tri); d=np.asarray(displacement,dtype=np.float64); h=np.asarray(targets,dtype=np.float64)
    if g.shape!=(*tri.shape,3) or tri.ndim!=2 or tri.shape[1]!=3 or h.shape!=(len(tri),) or not all(np.isfinite(x).all() for x in (g,d,h)): raise ValueError("invalid progressive rows")
    if not np.isfinite(trust_radius_mm) or trust_radius_mm<=0 or not 0<fraction<=1: raise ValueError("invalid trust policy")
    current=np.einsum("mij,mij->m",g,d[tri])
    support=trust_radius_mm*np.linalg.norm(g,axis=2).sum(axis=1)
    return current+fraction*np.minimum(np.maximum(0,h-current),support)
