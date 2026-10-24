"""Capacity-only conservative broadphase; not a complete obstacle-field clip."""
import numpy as np
from scipy.spatial import cKDTree
from stream_clip_009 import TetBroadphase

class CapacityBroadphase(TetBroadphase):
 def __init__(self, points, tets, fixed, target):
  full=TetBroadphase(points,tets)
  self.xyz=full.xyz
  f=np.asarray(fixed,bool)
  if f.shape!=(len(points),) or not np.isfinite(target) or not 0<target<1:
   raise ValueError("invalid fixed mask or target")
  tf=f[np.asarray(tets,np.int64)]
  eligible=np.flatnonzero(tf.any(1))
  self.lo=np.full((len(tets),3),np.inf);self.hi=np.full((len(tets),3),-np.inf)
  # Unsafe capacity region is the convex hull of fixed corners and fixed/free
  # edge intersections at capacity=target. Unfixed-only tets have capacity 1.
  for a in range(4):
   rows=np.flatnonzero(tf[:,a])
   self.lo[rows]=np.minimum(self.lo[rows],self.xyz[rows,a])
   self.hi[rows]=np.maximum(self.hi[rows],self.xyz[rows,a])
   for b in range(4):
    rows=np.flatnonzero(tf[:,a]&~tf[:,b])
    p=(1-target)*self.xyz[rows,a]+target*self.xyz[rows,b]
    self.lo[rows]=np.minimum(self.lo[rows],p);self.hi[rows]=np.maximum(self.hi[rows],p)
  self.lo[eligible]=np.nextafter(self.lo[eligible],-np.inf)
  self.hi[eligible]=np.nextafter(self.hi[eligible],np.inf)
  center=(self.lo[eligible]+self.hi[eligible])*.5
  radius=np.linalg.norm((self.hi[eligible]-self.lo[eligible])*.5,axis=1)
  bins=np.asarray([.00025,.0005,.001,.002,.004,.008,.016,.032,.064,.128,.256,1.,np.inf])
  labels=np.searchsorted(bins,radius,side='left');self.groups=[]
  for k in np.unique(labels):
   ii=np.flatnonzero(labels==k)
   self.groups.append((eligible[ii],cKDTree(center[ii]),float(radius[ii].max())))
  self.eligible=eligible
