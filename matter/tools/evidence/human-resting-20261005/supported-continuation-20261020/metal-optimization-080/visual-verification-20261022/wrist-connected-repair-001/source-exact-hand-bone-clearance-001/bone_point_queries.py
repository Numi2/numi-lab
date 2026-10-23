"""Conservative point-to-triangle queries for CPU anatomical preparation.
Unsigned distances are not a solid-containment or physical contact test.
"""
import numpy as np
from scipy.spatial import cKDTree
def closest(p,tri):
 a=tri[:,0];b=tri[:,1];c=tri[:,2];ab=b-a;ac=c-a;ap=p[None,:]-a
 d00=np.sum(ab*ab,1);d01=np.sum(ab*ac,1);d11=np.sum(ac*ac,1);d20=np.sum(ap*ab,1);d21=np.sum(ap*ac,1)
 den=np.maximum(d00*d11-d01*d01,1e-30);v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;u=1-v-w
 q=a+v[:,None]*ab+w[:,None]*ac;d2=np.sum((q-p)**2,1);inside=(u>=0)&(v>=0)&(w>=0)
 bestq=np.where(inside[:,None],q,0);bestd=np.where(inside,d2,np.inf)
 for i,j in ((0,1),(1,2),(2,0)):
  x=tri[:,i];y=tri[:,j];ed=y-x;l2=np.sum(ed*ed,1);t=np.clip(np.sum((p[None,:]-x)*ed,1)/np.maximum(l2,1e-30),0,1);q=x+t[:,None]*ed;d2=np.sum((q-p)**2,1);take=d2<bestd;bestq[take]=q[take];bestd[take]=d2[take]
 return np.sqrt(bestd),bestq

class TrianglePointQuery:
    def __init__(self, triangles):
        self.triangles=np.asarray(triangles,dtype=np.float64)
        if self.triangles.ndim!=3 or self.triangles.shape[1:]!=(3,3) or len(self.triangles)==0 or not np.isfinite(self.triangles).all():
            raise ValueError('nonempty finite (n,3,3) triangle array required')
        self.centroids=self.triangles.mean(1)
        self.max_radius=float(np.max(np.linalg.norm(self.triangles-self.centroids[:,None,:],axis=2)))
        self.tree=cKDTree(self.centroids)
    def query(self, point):
        point=np.asarray(point,dtype=np.float64)
        if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('finite xyz point required')
        _,seed=self.tree.query(point)
        upper=float(closest(point,self.triangles[int(seed):int(seed)+1])[0][0])
        # Triangle inequality: every closer triangle's centroid is in this ball.
        radius=upper+self.max_radius+1e-10
        ids=np.array(sorted(self.tree.query_ball_point(point,radius)),dtype=np.int64)
        if not len(ids):raise RuntimeError('conservative candidate set was empty')
        distances,points=closest(point,self.triangles[ids]);k=int(np.argmin(distances))
        return float(distances[k]),points[k],int(ids[k]),len(ids)
