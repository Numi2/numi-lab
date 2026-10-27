"""Identify genuinely replaced triangles so unchanged baseline faces are not new work."""
from collections import Counter
import numpy as np
def oriented_key(tri):
 t=tuple(map(int,tri))
 if len(t)!=3 or len(set(t))!=3:raise ValueError("three distinct vertices required")
 return min(t,t[1:]+t[:1],t[2:]+t[:2])
def changed_triangle_masks(old,new):
 a=np.asarray(old);b=np.asarray(new)
 if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3:raise ValueError("matched Kx3 patches required")
 if a.dtype.kind not in"iu"or b.dtype.kind not in"iu":raise ValueError("integer indices required")
 ak=[oriented_key(t)for t in a];bk=[oriented_key(t)for t in b]
 if len(set(ak))!=len(ak)or len(set(bk))!=len(bk):raise ValueError("duplicate oriented triangle")
 sa,sb=set(ak),set(bk)
 return np.asarray([t not in sb for t in ak],dtype=bool),np.asarray([t not in sa for t in bk],dtype=bool)

