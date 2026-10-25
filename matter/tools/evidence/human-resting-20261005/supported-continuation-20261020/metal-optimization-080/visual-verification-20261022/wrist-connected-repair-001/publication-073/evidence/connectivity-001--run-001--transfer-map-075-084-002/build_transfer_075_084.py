from __future__ import annotations
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, save_npz

HERE=Path(__file__).resolve().parent
BASE=Path("/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded")
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
EXP=ROOT/"accepted-transfer-row-export-075/run-001/source-preflight"
CAND=BASE/"connectivity-001/run-001/candidate-tetrahedra-compacted.npz"
MARKERS=BASE/"facet-marker-map.csv"
ADAPTER_DIR=ROOT/"E/accepted-transfer-row-adapter-075"
DETAIL=ROOT/"bilateral-hand-functional-rebind-020/run-001/candidate.nhskin"
PINS={
 "transfer_rows":"71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14",
 "coarse_reference":"9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2",
 "coarse_faces":"7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2",
 "detail_payload":"7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb",
}
PARENT_TOL_M=2e-7
BARY_TOL=2e-6
AREA_REL_TOL=2e-6

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as stream:
  for block in iter(lambda:stream.read(1<<20),b""): h.update(block)
 return h.hexdigest()

def csvrows(path):
 with Path(path).open(newline="",encoding="utf-8") as stream: return list(csv.DictReader(stream))

def barycentric(point, tri):
 a,b,c=tri
 v0=b-a; v1=c-a; v2=point-a
 d00=float(v0@v0); d01=float(v0@v1); d11=float(v1@v1)
 den=d00*d11-d01*d01
 if not np.isfinite(den) or den<=1e-30: return None,float("inf")
 d20=float(v2@v0); d21=float(v2@v1)
 u=(d11*d20-d01*d21)/den
 v=(d00*d21-d01*d20)/den
 weights=np.array([1-u-v,u,v],dtype=np.float64)
 return weights,float(np.linalg.norm(weights@tri-point))

def area(tri):
 return .5*float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0])))

def perm_sign(a,b):
 a=list(map(int,a)); b=list(map(int,b))
 if sorted(a)!=sorted(b): return 0
 p=[a.index(x) for x in b]
 inversions=sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
 return -1 if inversions%2 else 1

def main():
 t0=time.time()
 expected={
  "candidate":"5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db",
  "marker_map":"8e7ac9430c037d29e74d4c2f7f9f838995f6541c0f5fdf29a7b7095d9a6d8ff7",
  "adapter":"6f1c8887ab33d75bf5dd6241ec7a4a4ae8bc21d2d7573a8ba36b417f303c1117",
  **PINS,
 }
 paths={"candidate":CAND,"marker_map":MARKERS,"adapter":ADAPTER_DIR/"load_transfer_075.py",
  "transfer_rows":EXP/"accepted-transfer-rows.csv","coarse_reference":EXP/"accepted-coarse-reference.csv",
  "coarse_faces":EXP/"accepted-coarse-faces.csv","detail_payload":DETAIL}
 actual={k:sha(v) for k,v in paths.items()}
 for k,h in expected.items():
  if actual[k]!=h: raise RuntimeError(f"{k} SHA-256 mismatch: {actual[k]} != {h}")
 sys.path.insert(0,str(ADAPTER_DIR))
 from load_transfer_075 import load_transfer075
 W0,detail_origin,coarse_origin,_=load_transfer075(
  paths["transfer_rows"],paths["coarse_reference"],paths["coarse_faces"],paths["detail_payload"],
  expected_sha256=PINS)
 transfer_rows=csvrows(paths["transfer_rows"])
 face_rows=csvrows(paths["coarse_faces"])
 marker_rows=csvrows(MARKERS)
 z=np.load(CAND,allow_pickle=False)
 R=np.asarray(z["reference_points_m"],dtype=np.float64)
 current=np.asarray(z["current037_points_m_f32"],dtype=np.float32)
 source_input=np.asarray(z["source_input_row"],dtype=np.int64)
 output_ids=np.asarray(z["output_node_ids"],dtype=np.int64)
 child_faces=np.asarray(z["surface_faces_zero_based"],dtype=np.int64)
 child_markers=np.asarray(z["surface_face_markers"],dtype=np.int64)
 source_O=np.asarray(z["source_O_faces"],dtype=np.int64)
 if R.shape!=(7360,3) or current.shape!=R.shape or child_faces.shape!=(len(child_markers),3):
  raise RuntimeError("compact candidate shape/schema changed")
 source_rows=np.flatnonzero(source_input>=0)
 if len(source_rows)!=7233 or len(np.unique(source_input[source_rows]))!=7233 or source_input[source_rows].min()!=0 or source_input[source_rows].max()!=7232:
  raise RuntimeError("candidate does not retain unique original 7233-node domain")
 input_to_output=np.full(7233,-1,dtype=np.int64)
 input_to_output[source_input[source_rows]]=source_rows
 if np.any(input_to_output<0) or not np.array_equal(input_to_output[:2411],np.arange(2411)):
  raise RuntimeError("candidate original coarse O domain is not output rows 0..2410")
 coarse_faces=np.asarray([[int(r[f"vertex{i}"]) for i in range(3)] for r in face_rows],dtype=np.int64)
 stable_ids=np.asarray([int(r["stable_source_face_id"]) for r in face_rows],dtype=np.int64)
 if source_O.shape!=coarse_faces.shape or not np.array_equal(source_O,coarse_faces):
  raise RuntimeError("ordered candidate source_O face rows do not exactly equal frozen 075 coarse faces")
 # Origin positions and actual 075 selected face row identity are separately pinned.
 origin_residual=float(np.max(np.abs(R[input_to_output[:2411]]-coarse_origin.astype(np.float64))))
 if origin_residual!=0.0:
  raise RuntimeError(f"084 reference O prefix differs from 075 coarse reference F32 points: {origin_residual} m")
 # Map candidate child marker identities to exact original parent face rows.
 marker_info={}
 markers_by_parent=[[] for _ in range(len(coarse_faces))]
 for r in marker_rows:
  if r["surface"]!="O": continue
  marker=int(r["marker"]); parent=int(r["source_face_row"])
  vertices=np.fromstring(r["vertex_ids_zero_based"],sep=" ",dtype=np.int64)
  if vertices.shape!=(3,) or parent<0 or parent>=len(coarse_faces):
   raise RuntimeError("malformed O marker-map row")
  if not np.array_equal(np.sort(vertices),np.sort(source_O[parent])):
   raise RuntimeError(f"marker {marker} parent source row does not identify exact source_O face")
  if marker in marker_info: raise RuntimeError("duplicate O marker ID")
  marker_info[marker]=(parent,vertices)
  markers_by_parent[parent].append(marker)
 children_by_parent=[[] for _ in range(len(coarse_faces))]
 parent_area=np.empty(len(coarse_faces),dtype=np.float64)
 child_area_sum=np.zeros(len(coarse_faces),dtype=np.float64)
 parent_vertex_plane_max=np.zeros(len(coarse_faces),dtype=np.float64)
 parent_vertex_bary_min=np.full(len(coarse_faces),np.inf,dtype=np.float64)
 child_orientation_signs=[]
 marker_parent_order_signs=[]
 for p in range(len(coarse_faces)):
  parent_area[p]=area(R[input_to_output[source_O[p]]])
  if parent_area[p]<=0: raise RuntimeError(f"degenerate parent O face {p}")
  if not markers_by_parent[p]:
   raise RuntimeError(f"no O marker for source face row {p}")
  for marker in markers_by_parent[p]:
   marker_parent_order_signs.append(perm_sign(source_O[p],marker_info[marker][1]))
 for face_row,(tri_rows,marker) in enumerate(zip(child_faces,child_markers)):
  info=marker_info.get(int(marker))
  if info is None: continue
  parent,_marker_vertices=info
  tri=R[tri_rows]
  parent_tri=R[input_to_output[source_O[parent]]]
  parent_normal=np.cross(parent_tri[1]-parent_tri[0],parent_tri[2]-parent_tri[0])
  child_normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])
  ndot=float(parent_normal@child_normal)
  if ndot==0 or not np.isfinite(ndot): raise RuntimeError(f"degenerate child O face {face_row}")
  child_orientation_signs.append(1 if ndot>0 else -1)
  # Explicitly verify each child triangle remains inside the declared parent face.
  for point in tri:
   bary,res=barycentric(point,parent_tri)
   if bary is None: raise RuntimeError(f"degenerate parent O face {parent}")
   parent_vertex_plane_max[parent]=max(parent_vertex_plane_max[parent],res)
   parent_vertex_bary_min[parent]=min(parent_vertex_bary_min[parent],float(bary.min()))
  children_by_parent[parent].append(face_row)
  child_area_sum[parent]+=area(tri)
 # Number of O child faces is defined by marker surface labels, not a contiguous marker range.
 n_o_child=sum(len(group) for group in children_by_parent)
 if n_o_child!=4883 or any(len(g)==0 for g in children_by_parent):
  raise RuntimeError(f"candidate O partition incomplete: {n_o_child} O child faces")
 area_fraction=child_area_sum/parent_area
 max_area_error=float(np.max(np.abs(area_fraction-1.0)))
 max_plane=float(np.max(parent_vertex_plane_max))
 min_bary=float(np.min(parent_vertex_bary_min))
 # Parent source_O and accepted 075 table have identical winding; marker parent rows are reverse-wound;
 # emitted child O faces have the same geometric winding as source_O.
 coarse_to_source_sign=np.asarray([perm_sign(coarse_faces[i],source_O[i]) for i in range(len(source_O))])
 marker_to_source_sign=np.asarray(marker_parent_order_signs)
 child_sign=np.asarray(child_orientation_signs)
 if not np.all(coarse_to_source_sign==1): raise RuntimeError("075 and source_O parent face winding differs")
 if not np.all(marker_to_source_sign==-1): raise RuntimeError("marker parent winding relation changed; audit required")
 if not np.all(child_sign==1): raise RuntimeError("candidate O child winding disagrees with source_O parent orientation")
 if max_area_error>AREA_REL_TOL or max_plane>PARENT_TOL_M or min_bary < -BARY_TOL:
  raise RuntimeError(f"child O faces do not certify same-parent partition: area={max_area_error}, plane={max_plane}, minbary={min_bary}")
 for p in range(len(children_by_parent)):
  children_by_parent[p].sort(key=lambda fr:(int(stable_ids[p]),tuple(sorted(map(int,output_ids[child_faces[fr]]))),int(fr)))
 # Construct Wnew. The additive owner is detail_origin + Wnew @ (X - mechanical_origin_f32).
 # Wnew points are located in the pinned reference partition; current-map parity is an independent gate.
 n_detail,n_old=W0.shape
 if n_detail!=len(transfer_rows) or n_old!=2411: raise RuntimeError("unexpected 075 transfer shape")
 active=np.asarray([int(r["active"])==1 for r in transfer_rows],dtype=bool)
 W0f=W0.astype(np.float64)
 # Preserve the adapter's absolute W0 map for current/baseline diagnostics, but
 # do not use it for face membership: Float32 row-sum residue times the absolute
 # world translation can move a point slightly off its selected parent plane.
 old_reference_point_w0=np.asarray(W0f@R[input_to_output[:2411]],dtype=np.float64)
 old_current_origin=current[input_to_output[:2411]].astype(np.float64)
 old_current_point=np.asarray(W0f@old_current_origin,dtype=np.float64)
 rr=[]; cc=[]; vv=[]
 mapped_child=np.full(n_detail,-1,dtype=np.int64)
 mapped_parent=np.full(n_detail,-1,dtype=np.int64)
 mapped_residual_reference=np.full(n_detail,np.nan,dtype=np.float64)
 selected_parent_bary=np.full((n_detail,3),np.nan,dtype=np.float64)
 selected_parent_point=np.zeros((n_detail,3),dtype=np.float64)
 parent_bary_sum_error=np.full(n_detail,np.nan,dtype=np.float64)
 direct_rows=0; child_rows=0; tie_rows=0; max_f32_adjustment=0.0; failures=[]
 for i,row in enumerate(transfer_rows):
  if not active[i]: continue
  wr=W0.getrow(i)
  oldcols=wr.indices.astype(np.int64)
  oldweights=wr.data.astype(np.float64)
  if len(oldcols)==1 and abs(float(oldweights[0])-1.0)<=1e-7:
   col=int(oldcols[0]); output_row=int(input_to_output[col])
   if output_row<0: failures.append((i,"direct node has no output row")); continue
   rr.append(i); cc.append(output_row); vv.append(np.float32(1.0))
   selected_parent_point[i]=R[output_row]
   mapped_residual_reference[i]=0.0
   direct_rows+=1
   continue
  parent=int(row["selected_face_row"])
  if parent<0 or parent>=len(children_by_parent) or int(row["selected_face_stable_id"])!=int(stable_ids[parent]):
   failures.append((i,"selected source-face row/stable ID mismatch")); continue
  # Reconstruct affine coordinates in the exact selected source-face domain.
  # This removes only the measured Float32 partition-sum residue; it does not
  # project to a neighboring face or relax child containment tolerances.
  ordered_parent=np.asarray(source_O[parent],dtype=np.int64)
  face_bary=np.zeros(3,dtype=np.float64)
  valid_support=True
  for col,w in zip(oldcols,oldweights):
   where=np.flatnonzero(ordered_parent==int(col))
   if len(where)!=1:
    valid_support=False; break
   face_bary[int(where[0])]+=float(w)
  total=float(face_bary.sum())
  if not valid_support or not np.isfinite(total) or abs(total-1.0)>1.0e-6 or np.any(face_bary < -1.0e-8):
   failures.append((i,"075 transfer weights do not define the selected source-face affine point")); continue
  parent_bary_sum_error[i]=total-1.0
  face_bary/=total
  parent_tri=R[input_to_output[ordered_parent]]
  point=np.asarray(face_bary@parent_tri,dtype=np.float64)
  selected_parent_bary[i]=face_bary
  selected_parent_point[i]=point
  parent_plane_res=float(np.linalg.norm(point-barycentric(point,parent_tri)[0]@parent_tri))
  if parent_plane_res>1.0e-12:
   failures.append((i,f"selected-face affine reconstruction is off parent plane: {parent_plane_res}")); continue
  candidates=[]
  for child_row in children_by_parent[parent]:
   tri_rows=child_faces[child_row]
   tri=R[tri_rows]
   bary,res=barycentric(point,tri)
   if bary is None or res>PARENT_TOL_M or bary.min() < -BARY_TOL or bary.max()>1+BARY_TOL: continue
   clamped=np.maximum(bary,0.0)
   total=float(clamped.sum())
   if total<=0: continue
   clamped/=total
   res2=float(np.linalg.norm(clamped@tri-point))
   if res2>PARENT_TOL_M: continue
   key=(int(stable_ids[parent]),tuple(sorted(map(int,output_ids[tri_rows]))),int(child_row))
   candidates.append((key,int(child_row),tri_rows,clamped,res2))
  if not candidates:
   failures.append((i,"point not contained by any child of the exact selected parent face")); continue
  candidates.sort(key=lambda item:item[0])
  if len(candidates)>1: tie_rows+=1
  _,child_row,tri_rows,bary,res=candidates[0]
  weights32=bary.astype(np.float32)
  largest=int(np.argmax(weights32))
  before=weights32.copy()
  weights32[largest]=np.float32(float(weights32[largest])+(1.0-float(weights32.astype(np.float64).sum())))
  if np.any(weights32<0) or abs(float(weights32.astype(np.float64).sum())-1.0)>2e-7:
   failures.append((i,"Float32 child barycentric partition failed")); continue
  max_f32_adjustment=max(max_f32_adjustment,float(np.max(np.abs(weights32.astype(np.float64)-bary))))
  for col,w in zip(tri_rows,weights32):
   if w>0:
    rr.append(i);cc.append(int(col));vv.append(w)
  mapped_child[i]=child_row; mapped_parent[i]=parent
  mapped_residual_reference[i]=float(np.linalg.norm(weights32.astype(np.float64)@R[tri_rows]-point))
  child_rows+=1
 if failures: raise RuntimeError(f"{len(failures)} active transfer rows failed closed; first={failures[:12]}")
 Wnew=coo_matrix((np.asarray(vv,dtype=np.float32),(np.asarray(rr,dtype=np.int64),np.asarray(cc,dtype=np.int64))),
  shape=(n_detail,len(R)),dtype=np.float32).tocsr()
 Wnew.sum_duplicates();Wnew.eliminate_zeros();Wnew.sort_indices()
 row_sum=np.asarray(Wnew.sum(axis=1),dtype=np.float64).reshape(-1)
 if np.any(np.abs(row_sum[active]-1.0)>2e-7) or np.any(row_sum[~active]!=0.0): raise RuntimeError("Wnew partition or inactive zero rows failed")
 if not np.isfinite(Wnew.data).all() or (Wnew.nnz and Wnew.data.min()<0): raise RuntimeError("Wnew has invalid weights")
 # Exact input reference map is a face-local partition check; current map is the 037 displacement owner contract.
 new_reference_point=np.asarray(Wnew.astype(np.float64)@R,dtype=np.float64)
 new_current_point=np.asarray(Wnew.astype(np.float64)@current.astype(np.float64),dtype=np.float64)
 reference_diff=np.linalg.norm(new_reference_point-selected_parent_point,axis=1)
 current_diff=np.linalg.norm(new_current_point-old_current_point,axis=1)
 # Baseline additive deformation is exactly zero, so original detail origin is bitwise preserved.
 zero_delta=np.asarray(Wnew@(current-current),dtype=np.float32)
 baseline_detail_out=np.asarray(detail_origin+zero_delta,dtype=np.float32)
 if not np.array_equal(baseline_detail_out,detail_origin): raise RuntimeError("baseline additive map does not preserve detail origin bitwise")
 # New outer-node support report.
 o_node_rows=np.unique(np.concatenate([child_faces[fr] for group in children_by_parent for fr in group]))
 new_o_rows=o_node_rows[source_input[o_node_rows]<0]
 csc=Wnew.tocsc(); col_nnz=np.diff(csc.indptr)
 supported_new=new_o_rows[col_nnz[new_o_rows]>0]
 unsupported_new=new_o_rows[col_nnz[new_o_rows]==0]
 # Deterministic adjoint identity for the additive displacement map.
 rng=np.random.default_rng(75084)
 force=rng.standard_normal((n_detail,3)); displacement=rng.standard_normal((len(R),3))
 lhs=float(np.sum(force*(Wnew.astype(np.float64)@displacement)))
 rhs=float(np.sum((Wnew.astype(np.float64).T@force)*displacement))
 adj_abs=abs(lhs-rhs); adj_rel=adj_abs/max(abs(lhs),abs(rhs),1e-30)
 # 075 rest_offset remains diagnostic only; it is not used in the additive owner output.
 rows0=csvrows(paths["transfer_rows"])
 csv_offset=np.asarray([[float(r[f"rest_offset_{a}_m"]) for a in "xyz"] for r in rows0],dtype=np.float32)
 offset_diag=np.linalg.norm(csv_offset.astype(np.float64)[active]-
  (detail_origin.astype(np.float64)[active]-old_reference_point_w0[active]),axis=1)
 report={
  "schema":"075_to_compacted_084_additive_transfer_map_v3",
  "status":"pass_static_mapping_and_adjoint_checks",
  "inputs_sha256":actual,
  "candidate":{"node_count":int(len(R)),"tet_count":int(len(z["tetrahedra_zero_based"])),"surface_child_O_face_count":int(n_o_child)},
  "075":{"detail_rows":int(n_detail),"active_rows":int(active.sum()),"inactive_fixed_rows":int((~active).sum()),"coarse_nodes":int(n_old),"selected_parent_faces":int(len(coarse_faces))},
  "face_identity_and_orientation":{"ordered_075_faces_equal_source_O":True,"ordered_075_vs_source_O_winding_signs":{str(k):int(v) for k,v in zip(*np.unique(coarse_to_source_sign,return_counts=True))},"marker_parent_vs_source_O_winding_signs":{str(k):int(v) for k,v in zip(*np.unique(marker_to_source_sign,return_counts=True))},"candidate_child_normal_vs_source_O_parent_signs":{str(k):int(v) for k,v in zip(*np.unique(child_sign,return_counts=True))},"every_child_uses_marker_selected_source_parent":True,"all_4777_parent_faces_have_children":True,"max_child_area_fraction_closure_error":max_area_error,"max_child_vertex_parent_plane_residual_m":max_plane,"min_child_vertex_parent_barycentric":min_bary},
  "Wnew":{"shape":list(Wnew.shape),"nnz":int(Wnew.nnz),"direct_one_hot_rows":int(direct_rows),"exact_parent_child_barycentric_rows":int(child_rows),"multiple_containing_child_ties":int(tie_rows),"unmapped_active_rows":0,"nearest_fallback_used":False,"min_weight":float(Wnew.data.min()),"max_active_row_sum_error":float(np.max(np.abs(row_sum[active]-1.0))),"inactive_zero_rows_preserved":bool(np.all(row_sum[~active]==0)),"max_float32_bary_adjustment":max_f32_adjustment,"max_parent_affine_sum_error_before_normalization":float(np.nanmax(np.abs(parent_bary_sum_error[active]))),"max_raw_W0_to_affine_parent_point_residual_m":float(np.nanmax(np.linalg.norm(old_reference_point_w0[active]-selected_parent_point[active],axis=1)))},
  "additive_owner":{"formula":"detail_origin_f32 + Wnew @ (mechanical_origin_f32 - current_X_f32)","mechanical_origin_array":"candidate.current037_points_m_f32, exact full output-node domain","baseline_detail_origin_bitwise_preserved":True,"max_old_vs_new_reference_map_point_delta_m":float(reference_diff.max()),"rms_old_vs_new_reference_map_point_delta_m":float(np.sqrt(np.mean(reference_diff**2))),"max_old_vs_new_current037_map_point_delta_m":float(current_diff.max()),"rms_old_vs_new_current037_map_point_delta_m":float(np.sqrt(np.mean(current_diff**2))),"075_csv_rest_offset_used":False,"075_csv_rest_offset_active_formula_discrepancy_max_m":float(offset_diag.max()),"075_csv_rest_offset_active_formula_discrepancy_rms_m":float(np.sqrt(np.mean(offset_diag**2)))},
  "support":{"candidate_O_node_count":int(len(o_node_rows)),"new_O_node_count":int(len(new_o_rows)),"new_O_nodes_with_positive_transfer_support":int(len(supported_new)),"new_O_nodes_without_transfer_support":int(len(unsupported_new)),"unsupported_new_O_output_rows":unsupported_new.tolist()},
  "adjoint":{"lhs":lhs,"rhs":rhs,"absolute_error":adj_abs,"relative_error":adj_rel,"seed":75084},
  "scope":["Static W remap, source identity, face partition, and baseline additive-map checks only.","No solve, current geometry repair, detailed self-intersection audit, force/physics qualification, or native admission.","No nearest-face fallback; each remapped row is constrained to its exact 075 selected parent face."],
  "elapsed_seconds":time.time()-t0
 }
 save_npz(HERE/"Wnew-detail-to-084-nodes.npz",Wnew,compressed=True)
 np.savez_compressed(HERE/"transfer-sidecar.npz",
  detail_origin_f32=np.asarray(detail_origin,dtype=np.float32),
  mechanical_origin_f32=current,
  old_selected_parent_face_row=np.asarray([int(r["selected_face_row"]) for r in rows0],dtype=np.int64),
  old_selected_parent_stable_face_id=np.asarray([int(r["selected_face_stable_id"]) for r in rows0],dtype=np.int64),
  mapped_child_surface_face_row=mapped_child,mapped_parent_face_row=mapped_parent,
  mapped_point_residual_reference_m=mapped_residual_reference,
  old_parent_affine_barycentric=selected_parent_bary,
  old_parent_affine_point_reference_m=selected_parent_point,
  old_parent_affine_sum_error=parent_bary_sum_error,
  active_mask=active)
 report["outputs_sha256"]={p.name:sha(p) for p in [HERE/"Wnew-detail-to-084-nodes.npz",HERE/"transfer-sidecar.npz"]}
 report["outputs_sha256"]["builder"]=sha(Path(__file__))
 (HERE/"transfer-report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2))
if __name__=="__main__": main()
