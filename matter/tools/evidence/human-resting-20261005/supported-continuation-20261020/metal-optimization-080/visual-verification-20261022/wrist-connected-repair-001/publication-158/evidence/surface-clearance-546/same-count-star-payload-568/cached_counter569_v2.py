"""Reuse immutable bank AABBs for fixed-coordinate construction queries.
The same Float32->Float64 rigid transform and exact159 predicate are retained.
"""
import numpy as np
class CachedCounter:
 def __init__(self,original):
  self.original=original;self.data=[]
  for bank in original.banks:
   b=bank.triangles_local_m;lo=b.min(axis=1);hi=b.max(axis=1)
   self.data.append((bank,original._local_points(bank),lo,hi,lo.min(axis=0),hi.max(axis=0)))
 def count_faces(self,rows,faces):
  rows=np.asarray(rows);faces=np.asarray(faces)
  if rows.ndim!=1 or not len(rows) or not np.issubdtype(rows.dtype,np.integer):raise ValueError("nonempty integer rows required")
  if faces.shape!=(len(rows),3) or not np.issubdtype(faces.dtype,np.integer):raise ValueError("integer triangles required")
  if len(set(map(int,rows)))!=len(rows) or np.any(rows<0):raise ValueError("invalid rows")
  if np.any(faces<0) or np.any(faces>=len(self.original.points_f32)):raise ValueError("invalid vertex")
  result={}
  for bank,local,lo,hi,wholelo,wholehi in self.data:
   s=local[faces];sl=s.min(axis=1);sh=s.max(axis=1)
   active=np.flatnonzero(np.all(sh>=wholelo,axis=1)&np.all(sl<=wholehi,axis=1))
   empty={"count":0,"triangle_pairs":[],"aabb_candidate_pairs":0,"allowed_shared_vertex_or_edge_pairs":0}
   if not len(active):result[bank.key]=empty;continue
   blo=sl[active].min(axis=0);bhi=sh[active].max(axis=0)
   bi=np.flatnonzero(np.all(hi>=blo,axis=1)&np.all(lo<=bhi,axis=1))
   if not len(bi):result[bank.key]=empty;continue
   sr=self.original.records(s[active],rows[active].astype(np.int64,copy=False),self.original.exact)
   br=self.original.records(bank.triangles_local_m[bi],bank.stable_triangle_ids[bi],self.original.exact)
   audited=self.original.exact._audit_pair(sr,br,same_surface=False)
   pairs=audited.get("triangle_pairs")
   if not isinstance(pairs,list) or audited.get("count")!=len(pairs) or audited.get("allowed_shared_vertex_or_edge_pairs",0)!=0:raise ValueError("malformed exact result")
   valid_rows=set(map(int,rows));valid_triangle_ids=set(map(int,bank.stable_triangle_ids))
   for pair in pairs:
    if (not isinstance(pair,list) or len(pair)!=2 or type(pair[0]) is not int or type(pair[1]) is not int
        or pair[0] not in valid_rows or pair[1] not in valid_triangle_ids):raise ValueError("exact pair outside typed input domain")
   result[bank.key]={"count":len(pairs),"triangle_pairs":sorted(map(tuple,pairs)),
      "aabb_candidate_pairs":int(audited.get("aabb_candidate_pairs",0)),"allowed_shared_vertex_or_edge_pairs":0}
  expected={bank.key for bank in self.original.banks}
  if set(result)!=expected:raise ValueError("patch count did not cover every loaded typed bank")
  return result

