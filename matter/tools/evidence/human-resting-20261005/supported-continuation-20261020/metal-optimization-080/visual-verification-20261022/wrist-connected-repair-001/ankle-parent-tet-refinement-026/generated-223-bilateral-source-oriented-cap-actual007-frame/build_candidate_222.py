#!/usr/bin/env python3
import csv, hashlib, json, pathlib, collections
import numpy as np

BASE=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
E=BASE/"ankle-parent-tet-refinement-026"
GEN=BASE/"elbow-refinement-current-main-024-final080-candidate-source-frames-001"/"generated"
DUMP=BASE/"native-app-main-header-integration-151"/"source-preflight-007"/"initial-skin-fem-nodes.csv"
REG=BASE/"native-app-main-header-integration-151"/"source-preflight-007"/"registered-skin.csv"
PLACE=BASE/"elbow-refinement-current-main-024-final080-candidate-source-frames-001"/"native-source-only-024"/"common-rigid-transfer-row-diagnostic-005.json"
OUT=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/ankle-parent-tet-refinement-026/generated-223-bilateral-source-oriented-cap-actual007-frame")
OUT.mkdir(parents=True,exist_ok=False)

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read_csv(p):
 with open(p,newline="") as f: return list(csv.DictReader(f))
def canontri(t): return tuple(sorted(map(int,t)))
def edge_stats(F):
 count=collections.Counter(); orient=collections.Counter()
 for tri in F:
  a,b,c=map(int,tri)
  for u,v in ((a,b),(b,c),(c,a)):
   k=(min(u,v),max(u,v)); count[k]+=1; orient[k]+=1 if u<v else -1
 hist=collections.Counter(count.values())
 same=sum(1 for e,n in count.items() if n==2 and orient[e]!=0)
 return hist,same
def tri_area2(X,F):
 p=X[F]; return np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
def map_candidate_rows(z,actual_base,reg_by_id,side):
 nbase=len(actual_base)
 assert z["outer_world_m"].shape[0]>=nbase
 assert np.array_equal(np.flatnonzero(z["node_kind"]=="coarse"),np.arange(nbase)), (side,"coarse row layout")
 X=np.asarray(z["outer_world_m"],float)
 R=np.asarray(placement["placement_rotation"],float); t=np.asarray(placement["placement_translation_m"],float)
 Y=X@R.T+t
 Y[:nbase]=actual_base
 max_transform_error=float(np.max(np.linalg.norm((X[:nbase]@R.T+t)-actual_base,axis=1)))
 for i in range(nbase,len(Y)):
  k=str(z["node_kind"][i])
  if k=="source_vertex":
   vid=int(z["node_source_vertex_id"][i])
   Y[i]=reg_by_id[vid]
  elif k=="source_cut":
   a,b=map(int,z["node_source_edge_vertices"][i]); u=float(z["node_source_edge_t"][i])
   Y[i]=(1-u)*reg_by_id[a]+u*reg_by_id[b]
  elif k=="coarse_cut":
   a,b=map(int,z["node_coarse_edge_vertices"][i]); u=float(z["node_coarse_edge_t"][i])
   Y[i]=(1-u)*actual_base[a]+u*actual_base[b]
  elif k=="coarse_cut_split":
   a,b=map(int,z["node_parent_output_vertices"][i]); w=np.asarray(z["node_parent_weights"][i],float)
   assert 0<=a<len(Y) and 0<=b<len(Y) and abs(float(w.sum())-1)<1e-12
   Y[i]=w[0]*Y[a]+w[1]*Y[b]
  else: raise ValueError((side,i,k))
 deviation=np.linalg.norm(Y-(X@R.T+t),axis=1)
 return Y,max_transform_error,float(deviation.max()),float(np.quantile(deviation,.95))

def make_side(path,side,actual_base,reg_by_id):
 z=np.load(path)
 Y,base_err,rebased_max,rebased_p95=map_candidate_rows(z,actual_base,reg_by_id,side)
 kinds=np.asarray(z["face_kind"],int)
 assert len(kinds)==len(z["faces"])
 # Only newly authored cap/source/bridge facets are appended; untouched faces come from pinned current D36.
 patch_ids=np.flatnonzero(kinds!=0)
 assert set(np.unique(kinds[patch_ids]))=={1,2,3}
 faces=np.asarray(z["faces"][patch_ids],int).copy()
 # Preserve old D36 rows and remap only newly appended vertex indices.
 add_count=len(Y)-len(actual_base)
 row_map=np.arange(len(Y),dtype=np.int64)
 row_map[len(actual_base):]+=side_offset[side]
 faces=row_map[faces]
 meta={k:np.asarray(z[k][patch_ids]) for k in ("face_kind","parent_surface_face","detailed_source_face")}
 # Verify retained coarse faces are exact subset of the pinned D36 face table.
 retained={canontri(t) for t in np.asarray(z["faces"])[kinds==0]}
 assert len(retained)==int((kinds==0).sum())
 assert retained.issubset(base_face_set)
 return Y[ len(actual_base): ],faces,meta,retained,base_err,rebased_max,rebased_p95,int(add_count),z

nodes=read_csv(GEN/"nodes.csv")
assert len(nodes)==2411
base_ref_O=np.asarray([[float(r[k]) for k in ("ox","oy","oz")] for r in nodes],float)
faces_rows=read_csv(GEN/"faces.csv")
base_faces=np.asarray([[int(r[k]) for k in ("a","b","c")] for r in faces_rows],np.int64)
base_parent=np.asarray([int(r["parent_surface_face"]) for r in faces_rows],np.int64)
base_stable=np.asarray([int(r["stable_source_face"]) for r in faces_rows],np.int64)
assert base_faces.shape==(4777,3)
base_face_set={canontri(t) for t in base_faces}
# Exact native007 outer coordinates are the authoritative retained rows.
actual={}
with open(DUMP,newline="") as f:
 for r in csv.DictReader(f):
  if int(r["layer"])==0: actual[int(r["row"])]=[float(r[k]) for k in ("x_m","y_m","z_m")]
assert set(actual)==set(range(2411))
actual_base=np.asarray([actual[i] for i in range(2411)],float)
reg_by_id={}
with open(REG,newline="") as f:
 for r in csv.DictReader(f): reg_by_id[int(r["vertex"])]=np.asarray([float(r[k]) for k in ("wx","wy","wz")],float)
placement=json.load(open(PLACE))
assert placement["placement_rotation"] and placement["placement_translation_m"]
# Global side offsets are fixed before constructing either side.
p112=E/"generated-112-source-oriented-cap-refinement"/"candidate-112-source-oriented-cap-refinement.npz"
p113=E/"generated-113-right-source-oriented-cap-refinement"/"candidate-113-right-source-oriented-cap-refinement.npz"
z112=np.load(p112); z113=np.load(p113)
nbase=2411
side_offset={"left":0,"right":len(z112["outer_world_m"])-nbase}
left=make_side(p112,"left",actual_base,reg_by_id)
right=make_side(p113,"right",actual_base,reg_by_id)
XL,FL,ML,RL,errL,maxL,p95L,addL,zl=left
XR,FR,MR,RR,errR,maxR,p95R,addR,zr=right
assert not (base_face_set-RL)&(base_face_set-RR), "left/right selected coarse patches overlap"
keep=RL&RR
assert len(keep)==len(base_faces)-279-269, (len(keep),len(base_faces))
keep_rows=[i for i,t in enumerate(base_faces) if canontri(t) in keep]
assert len(keep_rows)==len(keep)
base_keep=base_faces[keep_rows]
# Concatenate exact actual007 old rows and side-specific source/cut rows.
X=np.vstack([actual_base,XL,XR])
F=np.vstack([base_keep,FL,FR])
K=np.r_[np.zeros(len(base_keep),np.uint8),ML["face_kind"].astype(np.uint8),MR["face_kind"].astype(np.uint8)]
S=np.r_[np.zeros(len(base_keep),np.uint8),np.ones(len(FL),np.uint8),np.full(len(FR),2,np.uint8)]
P=np.r_[base_parent[keep_rows],ML["parent_surface_face"].astype(np.int64),MR["parent_surface_face"].astype(np.int64)]
D=np.r_[np.full(len(base_keep),-1,np.int64),ML["detailed_source_face"].astype(np.int64),MR["detailed_source_face"].astype(np.int64)]
orig=np.r_[np.asarray(keep_rows,np.int64),np.full(len(FL)+len(FR),-1,np.int64)]
assert len(np.unique(np.sort(F,axis=1),axis=0))==len(F), "duplicate output triangles"
hist,same=edge_stats(F)
assert max(hist,default=0)<=2, hist
assert same==0, same
areas=tri_area2(X,F)
X32=X.astype(np.float32)
areas32=tri_area2(X32,F)
assert np.isfinite(X).all() and np.isfinite(areas).all()
assert float(areas.min())>0 and float(areas32.min())>0
# Old coordinates are preserved bit-for-bit from native007 for every old node.
assert np.array_equal(X[:nbase],actual_base)
out_npz=OUT/"candidate-222.npz"
np.savez_compressed(out_npz,outer_world_actual007_m=X,surface_faces=F,
 surface_face_kind=K,surface_face_source_side=S,surface_face_parent_surface_ids=P,
 surface_face_detailed_source_ids=D,surface_face_original_d36_row=orig,
 original_d36_outer_world_actual007_m=actual_base,
 left_added_outer_world_actual007_m=XL,right_added_outer_world_actual007_m=XR)
report={
 "schema":"numi.human.bilateral-source-oriented-cap-actual007-surface.v1",
 "status":"surface-construction-only; not a volumetric or clearance-qualified candidate",
 "inputs":{
  "candidate112_sha256":sha(p112),"candidate113_sha256":sha(p113),
  "native007_node_dump_sha256":sha(DUMP),"native007_registered_skin_sha256":sha(REG),
  "d36_nodes_sha256":sha(GEN/"nodes.csv"),"d36_faces_sha256":sha(GEN/"faces.csv"),
  "placement_json_sha256":sha(PLACE),"placement_rotation":placement["placement_rotation"],
  "placement_translation_m":placement["placement_translation_m"]},
 "frame_rebase":{
  "old_coarse_rows_per_candidate":nbase,"left_max_transformed_candidate_vs_native007_um":errL*1e6,
  "right_max_transformed_candidate_vs_native007_um":errR*1e6,
  "left_max_rebased_node_delta_vs_transformed_reference_um":maxL*1e6,
  "left_p95_rebased_node_delta_um":p95L*1e6,
  "right_max_rebased_node_delta_vs_transformed_reference_um":maxR*1e6,
  "right_p95_rebased_node_delta_um":p95R*1e6,
  "direct_source_vertices_and_cut_edges_reconstructed_from_registered_skin_csv":True,
  "coarse_cut_nodes_reconstructed_from_actual007_outer_and_edge_barycentrics":True},
 "topology":{
  "d36_original_surface_faces":len(base_faces),"left_retained_d36_faces":len(RL),
  "right_retained_d36_faces":len(RR),"unchanged_faces_after_bilateral_replacement":len(base_keep),
  "left_authored_cap_source_bridge_faces":len(FL),"right_authored_cap_source_bridge_faces":len(FR),
  "merged_surface_faces":len(F),"merged_outer_surface_nodes":len(X),
  "left_added_nodes":len(XL),"right_added_nodes":len(XR),
  "oriented_edge_incidence_histogram":dict(hist),"same_direction_two_face_edges":same,
  "triangle_min_area2_f64_m2":float(areas.min()),
  "triangle_min_area2_actual007_f32_m2":float(areas32.min()),
  "f32_zero_area_faces":int(np.sum(areas32==0)),
  "duplicate_triangles":0},
 "preservation":{
  "old_node_rows_exact_from_native007":True,
  "unmodified_d36_faces_bitwise_connectivity_preserved":True,
  "unmodified_d36_face_rows":len(keep_rows)},
 "not_yet_checked":[
  "nearby-bone clearance and full-triangle intersections in actual007 frame",
  "skin/fat nested common-field construction",
  "cavity-volume closure, positive child tetrahedra, conditioning",
  "mass and material-volume transfer",
  "FEM node-count suitability; the detailed source surface is an envelope, not a decision to promote every fine point to an independent FEM coordinate"],
 "candidate_sha256":None}
report["candidate_sha256"]=sha(out_npz)
(OUT/"report-222.json").write_text(json.dumps(report,indent=2)+"\n")
(OUT/"run-222-inputs.json").write_text(json.dumps({"candidate112":str(p112),"candidate113":str(p113),"source":str(pathlib.Path(__file__).resolve())},indent=2)+"\n")
print(json.dumps({"candidate":str(out_npz),"candidate_sha256":report["candidate_sha256"],"report":str(OUT/"report-222.json"),"nodes":len(X),"faces":len(F),"new_left":len(XL),"new_right":len(XR),"face_hist":dict(hist),"same_direction":same,"area2_f32_min":float(areas32.min()),"frame_error_um":[errL*1e6,errR*1e6],"rebase_delta_um":[maxL*1e6,maxR*1e6]},indent=2))
