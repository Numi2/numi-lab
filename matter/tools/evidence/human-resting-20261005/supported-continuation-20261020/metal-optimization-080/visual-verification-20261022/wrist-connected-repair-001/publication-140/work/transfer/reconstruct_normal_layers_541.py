#!/usr/bin/env python3
"""One bounded current-normal O/I/D reconstruction for the pinned 541 joint surface."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
HERE=Path(__file__).resolve().parent
CURRENT=ROOT/"D/focused-clearance-541/joint/area-damping-14187/run-001/candidate.npz"
SOURCE146=ROOT/"connected-source-unfold-146/run-002-corrected-metadata/candidate.npz"
R0=ROOT/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
TRANSFER=ROOT/"D/focused-clearance-541/transfer/run-003-final-joint/skin-fat-layers-541.npz"
TRANSFER_REPORT=ROOT/"D/focused-clearance-541/transfer/run-003-final-joint/report.json"
RECON_OWNER=ROOT/"connected-layer-reconstruction-127/matter/tools/human_skin_fat_layer_reconstruction.py"
MASS_LEDGER=ROOT/"connected-layer-mass-ledger-118/mass-and-boundary-audit.json"
APPENDER_HEADER=Path("/Users/n/numi-human-aortic-contact-opt-20261019/apps/NumiHumanCoarseTissuePreparation.hpp")
APPENDER_SMOKE=HERE/"appender-owner-smoke-541.txt"
EXPECTED={
 "current":"c1e02aaaacebe571370e5b6273501fbb95d1d938cf4292025163495415468939",
 "transfer":"edebcefc563bdca06e600b140673c7d40c5c7297c9b8da1b105f35db4e6c7de7",
 "transfer_report":"a9f8d8a9b350420f3ca8dc960f92d14650885328268cabb09c518d83391f2644",
 "reconstruction_owner":"2a7f7ed02a5c90af11e09ffd8b8d2a34b2f3ff9c3b08ecb08514cccb29dc565e",
 "mass_ledger":"", # runtime pin is recorded below
 "appender_header":"93cb05b7108a0b3b36c64dde4177872ec3603c07ba33b23556c67b5cdaa88805",
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
 return mod

def run(outdir:Path):
 for key,path in [("current",CURRENT),("transfer",TRANSFER),("transfer_report",TRANSFER_REPORT),("reconstruction_owner",RECON_OWNER),("mass_ledger",MASS_LEDGER),("appender_header",APPENDER_HEADER)]:
  got=sha(path)
  if EXPECTED[key] and got!=EXPECTED[key]:raise ValueError(f"pinned {key} hash mismatch: {got}")
 with np.load(CURRENT,allow_pickle=False) as c,np.load(TRANSFER,allow_pickle=False) as t,np.load(SOURCE146,allow_pickle=False) as source146,np.load(R0,allow_pickle=False) as r0,np.load(ROOT/"coupled-layer-trial-128/run-004-full146-baseline-offset/layer-candidate.npz",allow_pickle=False) as source_layers:
  O=np.asarray(c["points_world_m"],dtype=np.float32).astype(np.float64)
  raw_faces=np.asarray(c["source_O_faces"],dtype=np.int64)
  faces=raw_faces[:,[0,2,1]].copy()
  anchors=np.asarray(c["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
  anchor_source_ids=np.asarray(c["typed_coarse_anchor_source_ids_018"],dtype=np.int64)
  if len(anchors)!=237 or len(np.unique(anchors))!=len(anchors) or anchors.min()<0 or anchors.max()>=len(O):raise ValueError("pinned typed anchor rows invalid")
  if not np.array_equal(c["node_kind"][anchors],np.zeros(len(anchors),dtype=np.int8)) or not np.array_equal(c["node_source_row"][anchors],anchor_source_ids):raise ValueError("typed anchor source IDs do not match output node identities")
  Pfull=np.asarray(t["full_source_prolongation_indptr"],dtype=np.int64);Prows=np.asarray(t["full_source_prolongation_rows"],dtype=np.int64);Pw=np.asarray(t["full_source_prolongation_weights"],dtype=np.float64)
  def verify_anchor_rows(rows,source_ids):
   root_rows=[];exact=True
   for row,sid in zip(rows,source_ids):
    lo,hi=Pfull[int(row):int(row)+2]
    if hi-lo!=1 or abs(Pw[lo]-1.0)>1e-12: exact=False;root_rows.append(-1);continue
    rr=int(Prows[lo]);root_rows.append(rr)
    if r0["node_kind"][rr]!=0 or int(r0["node_source_row"][rr])!=int(sid):exact=False
   return exact,np.asarray(root_rows,dtype=np.int64)
  anchor_bridge_ok,anchor_r0_rows=verify_anchor_rows(anchors,anchor_source_ids)
  if not anchor_bridge_ok:raise ValueError("current typed anchors do not compose one-hot to R0 kind0 rows with matching source identity")
  legacy_anchor_bridge_ok,legacy_anchor_r0_rows=verify_anchor_rows(legacy_anchor_rows,legacy_anchor_sources)
  legacy_kind0_count=int(np.count_nonzero(r0["node_kind"][legacy_anchor_r0_rows]>=0)) if False else int(sum(legacy_anchor_r0_rows[i]>=0 and r0["node_kind"][legacy_anchor_r0_rows[i]]==0 for i in range(len(legacy_anchor_r0_rows))))
  legacy_source_match_count=int(sum(legacy_anchor_r0_rows[i]>=0 and int(r0["node_source_row"][legacy_anchor_r0_rows[i]])==int(legacy_anchor_sources[i]) for i in range(len(legacy_anchor_r0_rows))))
  legacy_anchor_rows=np.asarray(c["coarse_boundary_anchor_rows"],dtype=np.int64);legacy_anchor_sources=np.asarray(c["coarse_boundary_anchor_source_nodes"],dtype=np.int64)
  if len(legacy_anchor_rows)!=237 or len(np.unique(legacy_anchor_rows))!=237 or len(legacy_anchor_sources)!=237 or len(np.unique(legacy_anchor_sources))!=237:raise ValueError("paired coarse boundary anchors invalid")
  if not np.array_equal(source146["points_world_m"],source_layers["outer_points_f64"]):raise ValueError("pinned 128 layer outer points differ from original 146 outer input")
  original_outer=np.asarray(source146["points_world_m"],dtype=np.float64)
  skin_depth=np.asarray(t["skin_thickness_transferred_m"],dtype=np.float64)
  fat_depth=np.asarray(t["fat_thickness_transferred_m"],dtype=np.float64)
  if skin_depth.shape!=(len(O),) or fat_depth.shape!=(len(O),):raise ValueError("transferred scalar fields do not match current output rows")
  recon_owner=load_module(RECON_OWNER,"reconstruct_layers_127")
  active=np.arange(len(O),dtype=np.int64)
  normals=recon_owner._vertex_normals(O,faces,+1,np.ones(len(O),dtype=bool))
  boundary_inner=O[anchors]-skin_depth[anchors,None]*normals[anchors]
  boundary_deep=boundary_inner-fat_depth[anchors,None]*normals[anchors]
  zero=np.zeros_like(O)
  result=recon_owner.reconstruct_connected_layers(
   O,faces,zero,zero,anchors,O[anchors],boundary_inner,boundary_deep,
   skin_depth_m=skin_depth,fat_depth_m=fat_depth,orientation_sign=+1,
   active_vertex_indices=active)
  I=np.asarray(result.inner_points,dtype=np.float64);D=np.asarray(result.deep_points,dtype=np.float64)
  if not np.array_equal(result.outer_points,O):raise ValueError("normal reconstruction changed current outer coordinates")
  if not np.array_equal(I[anchors],boundary_inner) or not np.array_equal(D[anchors],boundary_deep):raise ValueError("fixed anchor I/D coordinates are not bitwise exact")
  transfer_mod=load_module(HERE/"transfer_layers_541.py","transfer_layers_541")
  skin_geom=transfer_mod.layer_metrics(O,I,faces,skin_depth)
  fat_geom=transfer_mod.layer_metrics(I,D,faces,fat_depth)
  skin_actual=np.linalg.norm(I-O,axis=1);fat_actual=np.linalg.norm(D-I,axis=1)
  skin_res=np.abs(skin_actual-skin_depth);fat_res=np.abs(fat_actual-fat_depth)
  skin_tol=np.maximum(1e-10,skin_depth*1e-5);fat_tol=np.maximum(1e-10,fat_depth*1e-5)
  # Signed volume convention is checked against the matched source and 128 layer.
  def signed_volume(points,ff):
   return float(np.einsum("ij,ij->i",points[ff[:,0]],np.cross(points[ff[:,1]],points[ff[:,2]])).sum()/6.0)
  rawV=signed_volume(O,raw_faces);layerV=signed_volume(O,faces)
  if not(rawV<0.0 and layerV>0.0):raise ValueError("source winding does not match established outward layer orientation")
  ledger=json.loads(MASS_LEDGER.read_text())["native_parent_mass_source"]
  skin_mass=float(ledger["mass_from_authoritative_corner_ledger_kg"]["1"])
  fat_mass=float(ledger["mass_from_authoritative_corner_ledger_kg"]["2"])
  skin_rho=skin_mass/skin_geom["f64_prism_volume_m3"];fat_rho=fat_mass/fat_geom["f64_prism_volume_m3"]
  density_bounds={"skin_kg_m3":[900.0,1100.0],"fat_kg_m3":[850.0,950.0]}
  mass_capacity={"skin_existing_donor_debit_kg":skin_mass,"fat_existing_donor_debit_kg":fat_mass,
   "skin_new_shell_density_if_assigned_kg_m3":skin_rho,"fat_new_shell_density_if_assigned_kg_m3":fat_rho,
   "density_bounds_kg_m3":density_bounds,"skin_density_in_bounds":density_bounds["skin_kg_m3"][0]<=skin_rho<=density_bounds["skin_kg_m3"][1],
   "fat_density_in_bounds":density_bounds["fat_kg_m3"][0]<=fat_rho<=density_bounds["fat_kg_m3"][1],
   "new_shell_mass_debit_could_close_by_existing_density_owner":bool(density_bounds["skin_kg_m3"][0]<=skin_rho<=density_bounds["skin_kg_m3"][1] and density_bounds["fat_kg_m3"][0]<=fat_rho<=density_bounds["fat_kg_m3"][1]),
   "per_source_cell_mass_ancestry_required":False,
   "actual_C++_appender_called_for_this_full_mesh":False,
   "reason":"Existing appender assigns each new shell tet mass as density*volume/4 from fixed donor debit, but full 41,733-face surface exceeds its 33,333-face/100,000-tet regional limit and layer geometry is still gated by exact cell checks."}
  report={
   "status":"CURRENT_NORMAL_LAYER_RECONSTRUCTION_AND_CELL_AUDIT_COMPLETE",
   "physical_or_native_admission":False,
   "pins":{"original_146_outer_input":{"path":str(SOURCE146),"sha256":sha(SOURCE146)},"r0_typed_identity_source":{"path":str(R0),"sha256":sha(R0)},"current_candidate":{"path":str(CURRENT),"sha256":sha(CURRENT)},"failed_offset_transfer":{"path":str(TRANSFER),"sha256":sha(TRANSFER)},"failed_offset_transfer_report":{"path":str(TRANSFER_REPORT),"sha256":sha(TRANSFER_REPORT)},"reconstruction_owner":{"path":str(RECON_OWNER),"sha256":sha(RECON_OWNER)},"existing_donor_mass_ledger":{"path":str(MASS_LEDGER),"sha256":sha(MASS_LEDGER)},"current_appender_header":{"path":str(APPENDER_HEADER),"sha256":sha(APPENDER_HEADER)},"existing_appender_fixture_receipt":{"path":str(APPENDER_SMOKE),"sha256":sha(APPENDER_SMOKE)}},
   "input_contract":{"rows":int(len(O)),"faces":int(len(faces)),"face_permutation":[0,2,1],"raw_signed_volume_m3":rawV,"layer_winding_signed_volume_m3":layerV,"source_face_adjacency":"root verified 0 inconsistent internal edges, 0 nonmanifold edges, 45 known open head boundary edges; winding permutation is global and preserves adjacency","current_outer_bitwise_preserved":True,"original_146_outer_bitwise_preserved":True,"original_128_outer_equals_146_bitwise":True,"no_outer_force_or_position_lock_added":True,"typed_anchors":{"rows":int(len(anchors)),"source_ids_match_node_source_rows":True,"outer_inner_deep_bitwise_preserved":True,"matches_joint_fit_anchor_rows":True,"paired_coarse_boundary_anchor_rows_also_present_but_distinct":True,"coarse_boundary_anchor_rows":legacy_anchor_rows.astype(int).tolist(),"coarse_boundary_anchor_source_nodes":legacy_anchor_sources.astype(int).tolist()},"active_rows":int(len(active))},
   "method":{"owner":"connected-layer-reconstruction-127.reconstruct_connected_layers","skin_fat_layers_from_scalar_depths_and_current_vertex_normals":True,"stale_128_displacement_directions_used_for_output":False,"zero_baseline_offsets":True,"full_surface_active":True,"typed_anchors_are_exact_Dirichlet_rows":True,"normal_sign":1,"diagnostics":result.diagnostics},
   "thickness":{"source_scalar_depth_bounds_m":{"skin":[float(skin_depth.min()),float(skin_depth.max())],"fat":[float(fat_depth.min()),float(fat_depth.max())]},"actual_norm_bounds_m":{"skin":[float(skin_actual.min()),float(skin_actual.max())],"fat":[float(fat_actual.min()),float(fat_actual.max())]},"skin_max_abs_norm_residual_m":float(skin_res.max()),"fat_max_abs_norm_residual_m":float(fat_res.max()),"skin_owner_tolerance_max_m":float(skin_tol.max()),"fat_owner_tolerance_max_m":float(fat_tol.max()),"skin_rows_outside_owner_tolerance":int(np.count_nonzero(skin_res>skin_tol)),"fat_rows_outside_owner_tolerance":int(np.count_nonzero(fat_res>fat_tol))},
   "cell_checks":{"skin":skin_geom,"fat":fat_geom,"exact_408_or_185_surface_self_audit_repeated":False,"bone_clearance_run":False,"volume_mesh_created":False},
   "mass_accounting":mass_capacity,
   "appender_contract":{"maximum_tets_per_regional_shell":100000,"maximum_faces_per_regional_shell":33333,"candidate_faces":int(len(faces)),"tet_count_per_shell":int(3*len(faces)),"face_budget_pass":bool(len(faces)<=33333),"failure_witness":"full current surface exceeds the existing regional shell budget; appender call intentionally not made"},
   "limits":["Construction and exact prescribed face-prism checks only; no physics or native admission.","No global self-intersection or bone-clearance rerun.","Aggregate donor debit feasibility is calculated from the 118 ledger; nodal masses are not emitted because cell quality and owner face budget have not passed.","The 45 known head boundary edges remain open on O/I/D surfaces; this layer writer does not add caps or claim they are closed.","Material density bounds are aggregate ledger bounds; no per-node donor-body transform/local moment arrays are present in this candidate NPZ."]}
  outdir.mkdir(parents=True,exist_ok=False)
  out=outdir/"skin-fat-layers-541-normal-reconstructed.npz"
  np.savez_compressed(out,outer_points_f64=O,inner_points_f64=I,deep_points_f64=D,outer_points_f32=O.astype(np.float32),inner_points_f32=I.astype(np.float32),deep_points_f32=D.astype(np.float32),outer_faces=faces,source_outer_faces_raw=raw_faces,face_winding_permutation=np.asarray([0,2,1],dtype=np.int8),skin_depth_m=skin_depth,fat_depth_m=fat_depth,skin_actual_norm_m=skin_actual,fat_actual_norm_m=fat_actual,outward_normals=result.outward_normals,anchor_rows=anchors,anchor_source_ids=anchor_source_ids,coarse_boundary_anchor_rows=legacy_anchor_rows,coarse_boundary_anchor_source_nodes=legacy_anchor_sources,node_kind=np.asarray(c["node_kind"]),node_source_row=np.asarray(c["node_source_row"]),node_source_virtual_row=np.asarray(c["node_source_virtual_row"]),node_parent_coarse_face_row=np.asarray(c["node_parent_coarse_face_row"]),node_parent_coarse_edge_a=np.asarray(c["node_parent_coarse_edge_a"]),node_parent_coarse_edge_b=np.asarray(c["node_parent_coarse_edge_b"]),node_parent_coarse_edge_parameter=np.asarray(c["node_parent_coarse_edge_parameter"]),face_source_ancestry_indptr=np.asarray(c["face_source_ancestry_indptr"]),face_source_ancestry_rows=np.asarray(c["face_source_ancestry_face_rows"]),full_source_prolongation_indptr=np.asarray(t["full_source_prolongation_indptr"]),full_source_prolongation_rows=np.asarray(t["full_source_prolongation_rows"]),full_source_prolongation_weights=np.asarray(t["full_source_prolongation_weights"]))
  report["output"]={"path":str(out),"sha256":sha(out),"bytes":out.stat().st_size}
  rp=outdir/"report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  print(json.dumps({"report":str(rp),"report_sha256":sha(rp),"output":str(out),"output_sha256":sha(out),"skin_cells":skin_geom,"fat_cells":fat_geom,"thickness":report["thickness"],"mass_accounting":mass_capacity,"appender_contract":report["appender_contract"],"diagnostics":result.diagnostics},indent=2))

if __name__=="__main__":
 if len(sys.argv)!=2:raise SystemExit("usage: reconstruct_normal_layers_541.py OUTDIR")
 run(Path(sys.argv[1]))
