"""Exact introduced triangle witnesses for construction damping; no admission."""
import numpy as np
def typed_pairs(report,face_count):
 if report.get("complete") is not True or report.get("bank_count")!=185 or report.get("expected_bank_count")!=185 or report.get("surface_face_count")!=face_count:
  raise ValueError("complete matched 185-bank surface report required")
 pairs=set();keys=set()
 for b in report["banks"]:
  k=(int(b["body"]),int(b["stable_bone_id"]))
  if k in keys:raise ValueError("duplicate typed bank")
  keys.add(k);r=b.get("result")
  if r is None:
   if b.get("skin_aabb_active")!=0:raise ValueError("unexplained missing narrowphase")
   continue
  rows=r.get("triangle_pairs")
  if not isinstance(rows,list) or r.get("count")!=len(rows):raise ValueError("incomplete crossing witnesses")
  for row in rows:
   if len(row)!=2 or any(not isinstance(i,int) or isinstance(i,bool)for i in row):raise ValueError("invalid typed triangle witness")
   f,t=row
   if not 0<=f<face_count or t<0:raise ValueError("triangle witness outside domain")
   p=(*k,f,t)
   if p in pairs:raise ValueError("duplicate typed crossing")
   pairs.add(p)
 if len(keys)!=185:raise ValueError("missing typed banks")
 return keys,pairs
def introduced_surface_faces(before,after,face_count):
 kb,pb=typed_pairs(before,face_count);ka,pa=typed_pairs(after,face_count)
 if kb!=ka or before.get("surface_point_count")!=after.get("surface_point_count"):raise ValueError("bank/topology identity mismatch")
 introduced=pa-pb
 return np.asarray(sorted({p[2]for p in introduced}),dtype=np.int64),len(introduced)
