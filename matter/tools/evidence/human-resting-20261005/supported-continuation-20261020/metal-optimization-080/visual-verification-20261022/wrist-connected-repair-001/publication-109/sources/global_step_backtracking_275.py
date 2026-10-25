"""Uniform connected construction-step damping; never a geometry certificate."""
import numpy as np
def reduce_global_step_scales(faces,scales,bad_faces,self_pairs,*,anchors,zero_below):
 F=np.asarray(faces);s=np.asarray(scales,dtype=float).copy();a=np.asarray(anchors,dtype=np.int64)
 if F.ndim!=2 or F.shape[1]!=3 or s.ndim!=1 or not np.isfinite(s).all() or np.any(s<0) or np.any(s>1) or np.any(F<0) or np.any(F>=len(s)):
  raise ValueError("invalid connected global step domain")
 if not 0<zero_below<1:raise ValueError("invalid construction cutoff")
 if np.any(a<0) or np.any(a>=len(s)):raise ValueError("invalid anchors")
 if len(bad_faces)==0 and len(self_pairs)==0:raise ValueError("actual failure witnesses required")
 old=s.copy();s*=.5;s[s<zero_below]=0;s[a]=0
 return s,dict(status="UNIFORM_CONSTRUCTION_STEP_HALVED",construction_only=True,geometry_certified=False,factor=.5,zero_below=zero_below,changed_nodes=int(np.count_nonzero(old!=s)),unaffected_scales_bitwise_unchanged=True)
