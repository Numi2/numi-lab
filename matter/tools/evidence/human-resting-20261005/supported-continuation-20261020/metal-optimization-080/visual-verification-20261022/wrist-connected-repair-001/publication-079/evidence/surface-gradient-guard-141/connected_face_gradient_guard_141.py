"""Sparse shape bounds for static connected mesh authoring; no runtime physics."""
import numpy as np
from scipy import sparse

def build_face_gradient_guard(points, faces, *, max_frobenius=0.4, condition_limit=1e8):
    P=np.asarray(points,dtype=np.float64);F=np.asarray(faces)
    if P.ndim!=2 or P.shape[1]!=3 or not np.isfinite(P).all():
        raise ValueError("finite N x 3 points required")
    if F.ndim!=2 or F.shape[1]!=3 or F.dtype.kind not in "iu":
        raise ValueError("integer M x 3 faces required")
    if F.size and (F.min()<0 or F.max()>=len(P)):
        raise ValueError("face index out of range")
    if not np.isfinite(max_frobenius) or not 0<max_frobenius<1:
        raise ValueError("max_frobenius must be in (0,1)")
    if not np.isfinite(condition_limit) or condition_limit<1:
        raise ValueError("invalid condition limit")
    if not len(F):
        return sparse.csr_matrix((0,3*len(P))),float(max_frobenius/np.sqrt(6)),dict(face_count=0,maximum_condition=0.0)
    E=np.stack((P[F[:,1]]-P[F[:,0]],P[F[:,2]]-P[F[:,0]]),axis=2)
    lengths=np.linalg.norm(E[:,:,0],axis=1)
    normals=np.cross(E[:,:,0],E[:,:,1]);areas=np.linalg.norm(normals,axis=1)
    if np.any(lengths<=0) or np.any(areas<=0):
        raise ValueError("degenerate reference face")
    t0=E[:,:,0]/lengths[:,None];n=normals/areas[:,None];t1=np.cross(n,t0)
    T=np.stack((t0,t1),axis=2)
    Dm=np.einsum("fji,fjk->fik",T,E)
    cond=np.linalg.cond(Dm)
    if not np.isfinite(cond).all() or np.any(cond>condition_limit):
        raise ValueError("ill-conditioned reference face")
    inv=np.linalg.inv(Dm)
    rr=[];cc=[];vv=[]
    for j in range(3):
        for k in range(2):
            rows=6*np.arange(len(F))+2*j+k
            for corner,coef in ((0,-inv[:,0,k]-inv[:,1,k]),(1,inv[:,0,k]),(2,inv[:,1,k])):
                rr.append(rows);cc.append(3*F[:,corner]+j);vv.append(coef)
    G=sparse.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(6*len(F),3*len(P))).tocsr()
    return G,float(max_frobenius/np.sqrt(6)),dict(face_count=int(len(F)),maximum_condition=float(cond.max()),maximum_displacement_gradient_frobenius=float(max_frobenius),static_construction_only=True)
