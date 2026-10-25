import hashlib,json
from pathlib import Path
import numpy as np

IN=Path("/Users/numi/Work/numi-human-connected-surface-069/connected-source-unfold-146/run-002-corrected-metadata/candidate.npz")
OUTDIR=Path("/Users/numi/Work/numi-human-connected-surface-069/connected-source-patch-cut-123/run-031-fixed-coordinate-edge-flips")
OUT=OUTDIR/"candidate.npz"; REPORT=OUTDIR/"flip-report.json"
EXPECTED="4a393af65cc12181e00f2b578f73ad209f87e5f2107f4808e9ce083f46a2b3da"
PAIRS=[(9185,9186),(10677,10678)]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def area_normal(X,t):
 q=X[np.asarray(t,dtype=np.int64)];n=np.cross(q[1]-q[0],q[2]-q[0]);v=float(np.linalg.norm(n))
 return .5*v,(n/v if v else np.zeros(3))
def edge_map(F):
 d={}
 for fi,t in enumerate(F):
  for a,b in zip(t,np.roll(t,-1)):
   key=tuple(sorted((int(a),int(b))))
   d.setdefault(key,[]).append((fi,int(a),int(b)))
 return d
def csr_rows(ptr,values,i): return np.asarray(values[int(ptr[i]):int(ptr[i+1])],dtype=np.int32)
def main():
 OUTDIR.mkdir(parents=True,exist_ok=True)
 if OUT.exists() or REPORT.exists(): raise FileExistsError("run-031 outputs exist")
 if sha(IN)!=EXPECTED: raise RuntimeError("input candidate SHA mismatch")
 z=np.load(IN,allow_pickle=False); d={k:np.asarray(z[k]).copy() for k in z.files}
 X=d["points_world_m"]; F=d["source_O_faces"].astype(np.int32,copy=True)
 if X.shape!=(8776,3) or F.shape!=(17507,3): raise RuntimeError("unexpected source mesh shape")
 oldF=F.copy(); edges=edge_map(F); operations=[]; flip_rows=[]
 src_ptr=d["face_source_ancestry_indptr"]; src_values=d["face_source_ancestry_face_rows"]
 sub_ptr=d["face_fine_subface_ancestry_indptr"]; sub_values=d["face_fine_subface_ancestry_rows"]
 face_src=[]; face_sub=[]
 for target,neighbor in PAIRS:
  ta=list(map(int,F[target])); nb=list(map(int,F[neighbor]))
  shared=sorted(set(ta)&set(nb))
  if len(shared)!=2: raise RuntimeError(f"faces {target}/{neighbor} do not share one edge")
  sa,sb=shared
  # Preserve the directed boundary of the two-face patch; the shared edge
  # is removed and replaced by the opposite diagonal.
  eA=next((int(u),int(v)) for u,v in zip(ta,ta[1:]+ta[:1]) if {int(u),int(v)}==set(shared))
  eB=next((int(u),int(v)) for u,v in zip(nb,nb[1:]+nb[:1]) if {int(u),int(v)}==set(shared))
  if eA!=eB[::-1]: raise RuntimeError(f"local winding mismatch at {target}/{neighbor}")
  cA=next(x for x in ta if x not in shared); cB=next(x for x in nb if x not in shared)
  new=[np.asarray([eA[0],cB,cA],dtype=np.int32),np.asarray([cB,eA[1],cA],dtype=np.int32)]
  old_stats=[area_normal(X,oldF[r]) for r in (target,neighbor)]
  new_stats=[area_normal(X,t) for t in new]
  if min(v for v,n in new_stats)<=1e-12: raise RuntimeError("replacement face below area floor")
  # The old near-collinear face normal is unstable; require each new face to
  # agree with the well-conditioned opposite face normal.
  ref=old_stats[1][1]
  dots=[float(np.dot(ref,n)) for v,n in new_stats]
  if min(dots)<0.999999: raise RuntimeError(f"replacement orientation mismatch {dots}")
  source_union=sorted(set(csr_rows(src_ptr,src_values,target).tolist()+csr_rows(src_ptr,src_values,neighbor).tolist()))
  if not source_union:
   source_union=sorted(set([int(d["parent_fine_detail_face_row"][target]),int(d["parent_fine_detail_face_row"][neighbor])]))
  sub_union=sorted(set(csr_rows(sub_ptr,sub_values,target).tolist()+csr_rows(sub_ptr,sub_values,neighbor).tolist()))
  if len(source_union)<2 or len(sub_union)<2: raise RuntimeError("flip must preserve both source and subface ancestry")
  F[target]=new[0]; F[neighbor]=new[1]
  flip_rows.extend([target,neighbor])
  face_src.append(source_union); face_sub.append(sub_union)
  operations.append({"face_rows":[target,neighbor],"old_faces":[ta,nb],"new_faces":[new[0].tolist(),new[1].tolist()],
   "shared_edge":[sa,sb],"new_diagonal":[cA,cB],"source_detail_ancestry_rows":source_union,"fine_subface_ancestry_rows":sub_union,
   "old_areas_m2":[old_stats[0][0],old_stats[1][0]],"new_areas_m2":[v for v,n in new_stats],"new_normal_dot_neighbor":[dots]})
  for row in (target,neighbor):
   for key in ["parent_coarse_face_row","parent_fine_detail_face_row","parent_source_face_marker","parent_fine_subface_row"]:
    d[key][row]=-1
 d["source_O_faces"]=F
 # Rebuild source-face and fine-subface ancestry CSR; only four flipped faces
 # change, and every new face conservatively carries the union of both parents.
 def rebuild(old_ptr,old_values,replace_map):
  ptr=np.zeros(len(F)+1,dtype=np.int64); vals=[]
  for i in range(len(F)):
   vals.extend(replace_map.get(i,csr_rows(old_ptr,old_values,i).tolist()))
   ptr[i+1]=len(vals)
  return ptr,np.asarray(vals,dtype=np.int32)
 src_map={r:face_src[j//2] for j,r in enumerate(flip_rows)}
 sub_map={r:face_sub[j//2] for j,r in enumerate(flip_rows)}
 d["face_source_ancestry_indptr"],d["face_source_ancestry_face_rows"]=rebuild(src_ptr,src_values,src_map)
 d["face_fine_subface_ancestry_indptr"],d["face_fine_subface_ancestry_rows"]=rebuild(sub_ptr,sub_values,sub_map)
 d["quality_edge_flip_face_rows"]=np.asarray(flip_rows,dtype=np.int32)
 d["quality_edge_flip_pair_face_rows"]=np.asarray(PAIRS,dtype=np.int32)
 d["quality_edge_flip_shared_edges"]=np.asarray([op["shared_edge"] for op in operations],dtype=np.int32)
 d["quality_edge_flip_new_diagonals"]=np.asarray([op["new_diagonal"] for op in operations],dtype=np.int32)
 d["quality_edge_flip_detail_ancestry_indptr"]=np.asarray([0]+[len(face_src[0]),len(face_src[0])+len(face_src[1])],dtype=np.int64)
 d["quality_edge_flip_detail_ancestry_rows"]=np.asarray(face_src[0]+face_src[1],dtype=np.int32)
 d["quality_edge_flip_subface_ancestry_indptr"]=np.asarray([0]+[len(face_sub[0]),len(face_sub[0])+len(face_sub[1])],dtype=np.int64)
 d["quality_edge_flip_subface_ancestry_rows"]=np.asarray(face_sub[0]+face_sub[1],dtype=np.int32)
 # Full-surface topology and orientation checks.
 em0=edge_map(oldF); em1=edge_map(F)
 if {e for e,v in em0.items() if len(v)==1}!={e for e,v in em1.items() if len(v)==1}: raise RuntimeError("boundary edge set changed")
 if max(map(len,em1.values()))>2: raise RuntimeError("nonmanifold edge created")
 winding_conflicts=sum(1 for e,vals in em1.items() if len(vals)==2 and (vals[0][1],vals[0][2])==(vals[1][1],vals[1][2]))
 old_conflicts=sum(1 for e,vals in em0.items() if len(vals)==2 and (vals[0][1],vals[0][2])==(vals[1][1],vals[1][2]))
 if winding_conflicts>old_conflicts: raise RuntimeError("edge flips increased directed winding conflicts")
 if not np.array_equal(d["points_world_m"],z["points_world_m"]): raise RuntimeError("coordinates changed")
 # Finite nondegenerate face areas are reported globally, not hidden by local checks.
 q=X[F]; cross=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]); areas=np.linalg.norm(cross,axis=1)*.5
 if not np.isfinite(areas).all(): raise RuntimeError("nonfinite area")
 with OUT.open("xb") as f: np.savez_compressed(f,**d)
 rep={"status":"FIXED_COORDINATE_PAIRED_EDGE_FLIPS_EMITTED","input":str(IN),"input_sha256":EXPECTED,
  "output":str(OUT),"output_sha256":sha(OUT),"source_rows_unchanged":True,"changed_face_rows":sorted(flip_rows),
  "operations":operations,"face_count":len(F),"vertex_count":len(X),"global_min_area_m2":float(areas.min()),
  "boundary_edge_count":sum(len(v)==1 for v in em1.values()),"nonmanifold_edge_count":sum(len(v)>2 for v in em1.values()),
  "directed_winding_conflicts_before":old_conflicts,"directed_winding_conflicts_after":winding_conflicts,
  "geometry_audit_status":"NOT_RUN","exact_self_gate_required":True,"bone_inside_191_unresolved":True}
 with REPORT.open("x") as f: json.dump(rep,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps(rep,sort_keys=True))
if __name__=="__main__":main()
