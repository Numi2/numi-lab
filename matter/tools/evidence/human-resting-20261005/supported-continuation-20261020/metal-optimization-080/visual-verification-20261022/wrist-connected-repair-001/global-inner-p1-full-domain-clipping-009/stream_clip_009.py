"""Conservative streamed triangle/tet broadphase for the existing 302 clipper.

Uses every current tetrahedron. No unsolved field or old tet domain can prune work.
"""
import numpy as np
from scipy.spatial import cKDTree

class TetBroadphase:
    def __init__(self, points, tets):
        x=np.asarray(points)
        t=np.asarray(tets)
        if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():
            raise ValueError("invalid points")
        if t.ndim!=2 or t.shape[1]!=4 or not np.issubdtype(t.dtype,np.integer) or np.any(t<0) or np.any(t>=len(x)):
            raise ValueError("invalid tetrahedra")
        self.xyz=x[t].astype(np.float64)
        self.lo=self.xyz.min(axis=1);self.hi=self.xyz.max(axis=1)
        c=self.xyz.mean(axis=1)
        radius=np.linalg.norm(self.xyz-c[:,None,:],axis=2).max(axis=1)
        # Radius bins change candidate cost only; each actual tet radius is bounded.
        bins=np.asarray([.00025,.0005,.001,.002,.004,.008,.016,.032,.064,.128,.256,1.,np.inf])
        labels=np.searchsorted(bins,radius,side="left")
        self.groups=[]
        for k in np.unique(labels):
            ids=np.flatnonzero(labels==k)
            upper=float(radius[ids].max())
            self.groups.append((ids,cKDTree(c[ids]),upper))
    def candidates(self, triangles):
        p=np.asarray(triangles,np.float64)
        if p.ndim!=3 or p.shape[1:]!=(3,3) or not np.isfinite(p).all():
            raise ValueError("invalid triangle batch")
        c=p.mean(axis=1);r=np.linalg.norm(p-c[:,None,:],axis=2).max(axis=1)
        lo=p.min(axis=1);hi=p.max(axis=1)
        out=[[] for _ in p]
        for ids,tree,upper in self.groups:
            rows=tree.query_ball_point(c,r+upper+2e-12)
            for i,local in enumerate(rows):
                if local:out[i].extend(ids[np.asarray(local,np.int64)].tolist())
        result=[]
        for i,rows in enumerate(out):
            ids=np.asarray(sorted(rows),np.int64)
            keep=np.all(self.hi[ids]>=lo[i]-2e-12,axis=1)&np.all(self.lo[ids]<=hi[i]+2e-12,axis=1)
            result.append(ids[keep])
        return result

def free_capacity(bary, tet_nodes, fixed):
    b=np.asarray(bary,np.float64)
    if b.ndim!=2 or b.shape[1]!=4 or not np.isfinite(b).all() or np.any(b<0) or (len(b) and np.max(np.abs(b.sum(axis=1)-1))>2e-12):
        raise ValueError("invalid clipped barycentric rows")
    return b[:,~fixed[np.asarray(tet_nodes,np.int64)]].sum(axis=1)
