#!/usr/bin/env python3
"""Transfer pinned 128 layer offsets through the exact 408 CSR lineage to 541."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
HERE=Path(__file__).resolve().parent
CURRENT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b61-97/root-layer-guard-002/backtrack-0p6/candidate-private.npz')
R0=ROOT/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
SOURCE146=ROOT/"connected-source-unfold-146/run-002-corrected-metadata/candidate.npz"
LAYERS=ROOT/"coupled-layer-trial-128/run-004-full146-baseline-offset/layer-candidate.npz"
LAYER_REPORT=ROOT/"coupled-layer-trial-128/run-004-full146-baseline-offset/report.json"
OWNER=ROOT/"publication-127/sources/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
EXPECTED={
 "current":"158e0dc7780ffa089efdd9d6adb4bfd0c066375c47a68d5820058e68716d91f8",
 "r0":"70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31",
 "source146":"4a393af65cc12181e00f2b578f73ad209f87e5f2107f4808e9ce083f46a2b3da",
 "layers":"301328142aae18bef98b1dfd9fff974533728aa5e084317d53fcda9dad5b7950",
 "owner":"5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2",
}

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def owner_compose(candidate, base):
 spec=importlib.util.spec_from_file_location("source_reference_408",OWNER)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 return mod.compose_source_reference(candidate,base)

def compose_matrix(c):
 prefixes=("","prior_input_006__","prior_run028__")
 n=len(c["points_world_m"]); P=sparse.eye(n,format="csr",dtype=np.float64); receipts=[]
 for level,prefix in enumerate(prefixes):
  ip=np.asarray(c[prefix+"source_prolongation_indptr"],dtype=np.int64)
  ix=np.asarray(c[prefix+"source_prolongation_node_rows"],dtype=np.int64)
  w=np.asarray(c[prefix+"source_prolongation_weights"],dtype=np.float64)
  nr=len(ip)-1
  nc=len(c[prefixes[level+1]+"source_prolongation_indptr"])-1 if level+1<len(prefixes) else 8776
  if nr!=P.shape[1] or ip.ndim!=1 or ip[0]!=0 or ip[-1]!=len(ix) or len(ix)!=len(w): raise ValueError(f"invalid CSR shape at level {level}")
  if np.any(np.diff(ip)<=0) or np.any(ix<0) or np.any(ix>=nc) or not np.isfinite(w).all() or np.any(w<0): raise ValueError(f"invalid CSR entries at level {level}")
  M=sparse.csr_matrix((w,ix,ip),shape=(nr,nc))
  err=float(np.max(np.abs(np.asarray(M.sum(axis=1)).ravel()-1)))
  if err>2e-10: raise ValueError(f"nonaffine CSR level {level}: {err}")
  P=(P@M).tocsr()
  receipts.append({"level":level,"prefix":prefix,"rows":nr,"columns":nc,"nnz":int(M.nnz),"max_row_sum_error":err})
 return P,receipts

def tetvol(a,b,c,d):
 return np.einsum("ij,ij->i",np.cross(b-a,c-a),d-a)/6.0

def area_min(points,faces):
 tri=np.asarray(points,dtype=np.float32)[faces].astype(np.float64)
 ar=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
 return float(ar.min()),int(np.count_nonzero(ar<=0))

def layer_metrics(outer,inner,faces,field=None):
 # Mirrors the appender's deterministic sorted-face prism decomposition.
 src=np.asarray(faces,dtype=np.int64); order=np.argsort(src,axis=1); f=np.take_along_axis(src,order,axis=1)
 inv=((src[:,0]>src[:,1]).astype(np.int8)+(src[:,0]>src[:,2]).astype(np.int8)+(src[:,1]>src[:,2]).astype(np.int8))
 parity=(inv&1).astype(bool); a,b,c=f.T
 v0=tetvol(outer[a],outer[b],outer[c],inner[c])
 v1=tetvol(outer[a],outer[b],inner[b],inner[c])
 v2=tetvol(outer[a],inner[a],inner[b],inner[c])
 thick=np.linalg.norm(inner-outer,axis=1)
 scalar=thick if field is None else np.asarray(field,dtype=np.float64)
 if scalar.shape!=(len(outer),) or not np.isfinite(scalar).all(): raise ValueError("invalid layer thickness field")
 tol=np.maximum(1e-8,np.maximum.reduce([scalar[a],scalar[b],scalar[c]]))**3*1e-12
 expected=np.where(parity,1.0,-1.0)
 authored=(np.isfinite(v0)&np.isfinite(v1)&np.isfinite(v2)&(np.abs(v0)>tol)&(np.abs(v1)>tol)&(np.abs(v2)>tol)&
           (np.sign(v0)==expected)&(np.sign(v1)==-expected)&(np.sign(v2)==expected))
 ids=np.stack([np.stack([a,b,c,len(outer)+c],axis=1),np.stack([a,b,len(outer)+b,len(outer)+c],axis=1),np.stack([a,len(outer)+a,len(outer)+b,len(outer)+c],axis=1)],axis=1).reshape(-1,4)
 xyz64=np.concatenate([np.asarray(outer,dtype=np.float64),np.asarray(inner,dtype=np.float64)],axis=0)
 q=xyz64[ids]; v64=np.einsum("ij,ij->i",np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])/6
 reverse=v64<0
 if np.any(reverse): ids[reverse,0],ids[reverse,1]=ids[reverse,1].copy(),ids[reverse,0].copy()
 xyz32=np.concatenate([np.asarray(outer,dtype=np.float32).astype(np.float64),np.asarray(inner,dtype=np.float32).astype(np.float64)],axis=0)
 p=xyz32[ids]
 vols=np.einsum("ij,ij->i",np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),p[:,3]-p[:,0])/6
 longest=np.zeros(len(p))
 for i in range(4):
  for j in range(i+1,4): longest=np.maximum(longest,np.linalg.norm(p[:,i]-p[:,j],axis=1))
 ndet=6*vols/np.maximum(longest,1e-300)**3
 amin,abad=area_min(inner,src); aout,aoutbad=area_min(outer,src)
 return {"face_count":len(src),"authored_prism_orientation_fail_count":int(np.count_nonzero(~authored)),"first_bad_face_rows":np.flatnonzero(~authored)[:20].astype(int).tolist(),
  "tet_count":int(len(vols)),"positive_f64_tet_count":int(np.count_nonzero(np.stack([v0,v1,v2],axis=1)>0)),
  "nonpositive_f32_tet_count":int(np.count_nonzero(vols<=0)),"f32_normalized_det_min":float(ndet.min()),
  "f32_conditioned_fail_count_lt_1e-5":int(np.count_nonzero(ndet<1e-5)),"f32_min_tet_volume_m3":float(vols.min()),
  "f64_prism_volume_m3":float(np.abs(np.stack([v0,v1,v2],axis=1)).sum()),"f32_outer_area_min_m2":aout,"f32_outer_area_nonpositive_count":aoutbad,"f32_area_min_m2":amin,
  "f32_area_nonpositive_count":abad,"actual_thickness_min_m":float(thick.min()),"actual_thickness_max_m":float(thick.max())}

def run(outdir:Path):
 for n,p in [("current",CURRENT),("r0",R0),("source146",SOURCE146),("layers",LAYERS),("owner",OWNER)]:
  got=sha(p)
  if got!=EXPECTED[n]: raise ValueError(f"pinned {n} hash mismatch: {got}")
 with np.load(CURRENT,allow_pickle=False) as c,np.load(R0,allow_pickle=False) as r,np.load(SOURCE146,allow_pickle=False) as a,np.load(LAYERS,allow_pickle=False) as l:
  typed_keys=("node_kind","node_source_row","node_source_virtual_row","node_parent_coarse_face_row","node_parent_coarse_edge_a","node_parent_coarse_edge_b","node_parent_coarse_edge_parameter")
  for key in typed_keys:
   if not np.array_equal(a[key],r[key],equal_nan=True): raise ValueError(f"146/R0 typed node provenance mismatch: {key}")
  for key in ("coarse_boundary_edge_endpoints","coarse_boundary_edge_path_indptr","coarse_boundary_edge_path_output_nodes","coarse_boundary_edge_path_source_nodes","coarse_boundary_edge_path_parameters","selected_coarse_face_rows","selected_fine_subface_rows","face_kind","parent_coarse_face_row"):
   if not np.array_equal(a[key],r[key],equal_nan=True): raise ValueError(f"146/R0 coarse edge/face provenance mismatch: {key}")
  if not np.array_equal(a["node_kind"],l["node_kind"]) or not np.array_equal(a["node_source_row"].astype(np.int64),l["node_source_row"].astype(np.int64)): raise ValueError("146 and 128 layer source node identity arrays differ")
  if not np.array_equal(np.sort(a["source_O_faces"],axis=1),np.sort(l["outer_faces"],axis=1)): raise ValueError("128 layer outer faces are not the same 146 triangles")
  P,levels=compose_matrix(c); base=np.asarray(r["points_before_joint_self_pair_repair_m"],dtype=np.float64); composed=np.asarray(P@base)
  owner_ref,owner_receipts=owner_compose(c,R0)
  if owner_ref.shape!=composed.shape or not np.allclose(owner_ref,composed,rtol=0,atol=2e-12): raise ValueError("408 composer result does not match explicitly multiplied full CSR lineage")
  old_o=np.asarray(l["outer_points_f64"],dtype=np.float64)
  source146_o=np.asarray(a["points_world_m"],dtype=np.float64)
  if old_o.shape!=source146_o.shape or not np.array_equal(old_o,source146_o): raise ValueError("128 layer outer points do not bitwise match its pinned 146 source")
  di=np.asarray(l["inner_points_f64"]-l["outer_points_f64"],dtype=np.float64)
  dd=np.asarray(l["deep_points_f64"]-l["inner_points_f64"],dtype=np.float64)
  skin_depth=np.asarray(l["skin_depth_m"],dtype=np.float64); fat_depth=np.asarray(l["fat_depth_m"],dtype=np.float64)
  skin_off=np.asarray(P@di); fat_off=np.asarray(P@dd)
  skin_field=np.asarray(P@skin_depth).ravel(); fat_field=np.asarray(P@fat_depth).ravel()
  O=np.asarray(c["points_world_m"],dtype=np.float32).astype(np.float64); I=O+skin_off; D=I+fat_off
  raw_faces=np.asarray(c["source_O_faces"],dtype=np.int64)
  raw_signed_volume=float(np.einsum("ij,ij->i",O[raw_faces[:,0]],np.cross(O[raw_faces[:,1]],O[raw_faces[:,2]])).sum()/6.0)
  faces=raw_faces[:,[0,2,1]].copy()
  oriented_signed_volume=float(np.einsum("ij,ij->i",O[faces[:,0]],np.cross(O[faces[:,1]],O[faces[:,2]])).sum()/6.0)
  if not(raw_signed_volume<0.0 and oriented_signed_volume>0.0): raise ValueError("face winding reversal does not match the pinned 128 outer-layer positive volume convention")
  if faces.shape!=(42379,3) or np.any(faces<0) or np.any(faces>=len(O)): raise ValueError("current faces invalid")
  if O.shape!=(P.shape[0],3) or len(skin_field)!=len(O): raise ValueError("current output row shape mismatch")
  sgeom=layer_metrics(O,I,faces,skin_field); fgeom=layer_metrics(I,D,faces,fat_field)
  sref=(float(np.min(l["skin_depth_m"])),float(np.max(l["skin_depth_m"])))
  fref=(float(np.min(l["fat_depth_m"])),float(np.max(l["fat_depth_m"])))
  report={
   "status":"LAYER_TRANSFER_AND_LOCAL_CELL_CHECK_COMPLETE","physical_or_native_admission":False,
   "pins":{"source_146_outer_input":{"path":str(SOURCE146),"sha256":sha(SOURCE146)},"current_candidate":{"path":str(CURRENT),"sha256":sha(CURRENT)},"r0_source":{"path":str(R0),"sha256":sha(R0)},"128_layer_source":{"path":str(LAYERS),"sha256":sha(LAYERS)},"128_report":{"path":str(LAYER_REPORT),"sha256":sha(LAYER_REPORT)},"408_composer":{"path":str(OWNER),"sha256":sha(OWNER)}},
   "csr_lineage":{"prefix_order":["","prior_input_006__","prior_run028__"],"levels":levels,"shape":[int(P.shape[0]),int(P.shape[1])],"nnz":int(P.nnz),"compose_source_reference_408_max_abs_match_m":float(np.max(np.abs(owner_ref-composed))),"owner_receipts":owner_receipts,"146_layer_outer_max_abs_match_m":float(np.max(np.abs(old_o-source146_o))),"146_R0_geometry_difference_max_abs_m":float(np.max(np.abs(source146_o-np.asarray(r["points_world_m"],dtype=np.float64))))},
   "face_winding":{"raw_source_signed_volume_m3":raw_signed_volume,"layer_convention_signed_volume_m3":oriented_signed_volume,"per_face_vertex_permutation":[0,2,1],"rationale":"Pinned 146/R0 source surface winding is inward (negative signed volume), while the matched 128 outer layer triangles use the opposite positive layer winding; reversing each face preserves adjacency and matches that layer convention."},
   "identity":{"verified_typed_node_keys":list(typed_keys),"146_R0_typed_node_provenance_equal":True,"146_R0_coarse_edge_and_face_keys_equal":["coarse_boundary_edge_endpoints","coarse_boundary_edge_path_indptr","coarse_boundary_edge_path_output_nodes","coarse_boundary_edge_path_source_nodes","coarse_boundary_edge_path_parameters","selected_coarse_face_rows","selected_fine_subface_rows","face_kind","parent_coarse_face_row"],"128_outer_triangles_match_146_unoriented":True,"current_output_rows":len(O),"current_output_faces":len(faces)},
   "transfer":{"method":"P from the exact 408 three-level source_prolongation CSR chain; apply P to original 128 I-O and D-I displacement vectors and scalar skin/fat depth fields, then reanchor at current 541 Float32 outer coordinates","outer_reanchored_to_current_candidate":True,"skin_depth_range_source_m":list(sref),"fat_depth_range_source_m":list(fref),"skin_depth_range_transferred_m":[float(skin_field.min()),float(skin_field.max())],"fat_depth_range_transferred_m":[float(fat_field.min()),float(fat_field.max())],"skin_actual_norm_range_m":[float(np.linalg.norm(I-O,axis=1).min()),float(np.linalg.norm(I-O,axis=1).max())],"fat_actual_norm_range_m":[float(np.linalg.norm(D-I,axis=1).min()),float(np.linalg.norm(D-I,axis=1).max())],"skin_actual_vs_transferred_depth_max_abs_m":float(np.max(np.abs(np.linalg.norm(I-O,axis=1)-skin_field))),"fat_actual_vs_transferred_depth_max_abs_m":float(np.max(np.abs(np.linalg.norm(D-I,axis=1)-fat_field)))},
   "cell_checks":{"skin":sgeom,"fat":fgeom,"exact_global_self_audit_run":False,"bone_clearance_run":False},
   "material_mass":{"conservative_cell_mass_transfer":False,"reason":"The 128 layer source has no volume-cell mesh or per-cell material/mass ancestry; the 541 ancestry arrays describe surface faces only and contain no volume fractions. No aggregate or invented density is represented as conservative transfer.","available_surface_face_ancestry_rows":int(len(c["face_source_ancestry_face_rows"])),"surface_face_ancestry_max_source_row":int(np.max(c["face_source_ancestry_face_rows"]))},
   "limits":["The output is a transferred geometric layer candidate and prescribed prism decomposition check only.","P@depth is a scalar interpolation check; actual layer thickness is the norm of transferred displacement and may differ from the scalar field.","No donor cell-volume partition or conservative mass ancestry was supplied, so physical FEM assembly and mass transfer are not qualified.","No global self-intersection, bone-clearance, TetGen, or native solver run was performed."]}
  outdir.mkdir(parents=True,exist_ok=False)
  out=outdir/"skin-fat-layers-541.npz"
  np.savez_compressed(out,outer_points_f64=O,inner_points_f64=I,deep_points_f64=D,outer_points_f32=O.astype(np.float32),inner_points_f32=I.astype(np.float32),deep_points_f32=D.astype(np.float32),outer_faces=faces,inner_faces=faces,deep_faces=faces,source_outer_faces_raw=raw_faces,face_winding_permutation=np.asarray([0,2,1],dtype=np.int8),skin_thickness_transferred_m=skin_field,fat_thickness_transferred_m=fat_field,skin_thickness_actual_norm_m=np.linalg.norm(I-O,axis=1),fat_thickness_actual_norm_m=np.linalg.norm(D-I,axis=1),node_kind=np.asarray(c["node_kind"]),node_source_row=np.asarray(c["node_source_row"]),source_face_ancestry_indptr=np.asarray(c["face_source_ancestry_indptr"]),source_face_ancestry_rows=np.asarray(c["face_source_ancestry_face_rows"]),node_source_virtual_row=np.asarray(c["node_source_virtual_row"]),node_parent_coarse_face_row=np.asarray(c["node_parent_coarse_face_row"]),node_parent_coarse_edge_a=np.asarray(c["node_parent_coarse_edge_a"]),node_parent_coarse_edge_b=np.asarray(c["node_parent_coarse_edge_b"]),node_parent_coarse_edge_parameter=np.asarray(c["node_parent_coarse_edge_parameter"]),full_source_prolongation_indptr=P.indptr.astype(np.int64),full_source_prolongation_rows=P.indices.astype(np.int64),full_source_prolongation_weights=P.data.astype(np.float64))
  report["output"]={"path":str(out),"sha256":sha(out),"bytes":out.stat().st_size}
  rp=outdir/"report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  print(json.dumps({"report":str(rp),"report_sha256":sha(rp),"output":str(out),"output_sha256":sha(out),"skin":sgeom,"fat":fgeom,"mass_transfer":report["material_mass"]},indent=2))

if __name__=="__main__":
 if len(sys.argv)!=2: raise SystemExit("usage: transfer_layers_541.py OUTDIR")
 run(Path(sys.argv[1]))
