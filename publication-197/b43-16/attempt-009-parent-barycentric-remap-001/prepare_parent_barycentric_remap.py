#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,importlib.util,sys
import numpy as np
HERE=Path(__file__).resolve().parent
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
B=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001"
C=B/"candidate.npz";P=S/"fit-from-663-001/seed-retry-663-001/fraction-small-001/candidate.npz"
A=B/"audit-159/report.json";SS=B/"audit-130/summary.json";SP=B/"audit-130/surface-self-pairs.npy"
O=Path("/Users/numi/Work/numi-human-connected-surface-069/D/local-two-parent-retriangulation-402/two_parent_patch_flips_402.py")
G=S/"edge-refinement-661-threebank-001/refresh_guards_multi_parent_generation002.py";AD=G.with_name("connected_refinement_adapter546_generation002.py")
R=HERE.parent/"attempt-008-cross-parent-flip-prep-001/ring-expansion-001/report.json";RPY=HERE.parent/"attempt-008-cross-parent-flip-prep-001/ring-expansion-001/one_ring_expansion_43_16.py"
OUT=HERE/"run-001";CP=OUT/"candidate.npz";REPORT=OUT/"report.json"
PIN={"candidate":"7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af",
"parent":"beaff5ca4deeba288662cbb614b1750516948c9400476618c6a9d45f823c9a35",
"audit159":"c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e",
"self_summary":"0ec99fa47245355d60858b05e1240b0e562de229c13721ea11659f6f4acdf5cf",
"self_pairs":"55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de",
"owner402":"f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764",
"generation002_guard":"90bafb6a51e9d7a2dd9b49570fcd977423616e18e992c2d4a4306feea5bd5eab",
"generation002_adapter":"6603e244496dfbbeb76f7cfbaa56c80f39b1a18e6cccb50727406148ba885fe4",
"ring_report":"80489c217f59688c192e9090c5d123a6b9bffe74ceff60205e02034732cdf0d5",
"ring_script":"df6087e13862147fbfbccf1d220fccea7a1ebb1a193813ef25c0d518b484b553"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loadmod(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
def loadnpz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
 preexisting_sha=sha(CP) if CP.is_file() else None
 if OUT.exists() and (not CP.is_file() or REPORT.exists()):raise FileExistsError(OUT)
 paths={"candidate":C,"parent":P,"audit159":A,"self_summary":SS,"self_pairs":SP,"owner402":O,"generation002_guard":G,"generation002_adapter":AD,"ring_report":R,"ring_script":RPY}
 got={k:sha(v) for k,v in paths.items()}
 if got!=PIN:raise ValueError(f"pinned input mismatch: {got}")
 ring=json.loads(R.read_text())
 if ring.get("candidate_written") is not False or ring.get("accepted_flip_count")!=1:raise ValueError("report-only ring proposal does not match expected single operation")
 c=loadnpz(C);parent=loadnpz(P);audit=json.loads(A.read_text());self_summary=json.loads(SS.read_text());selfpairs=np.load(SP,allow_pickle=False)
 if self_summary.get("candidate_sha256")!=PIN["candidate"] or self_summary.get("complete_surface_audit") is not True or self_summary.get("exact_PLC_surface_pairs")!=0 or selfpairs.shape!=(0,2):raise ValueError("stored baseline self receipt is not pinned zero-pair evidence")
 prep=loadmod("ring_prep_replay_for_bary_remap",RPY);owner=loadmod("owner402_for_bary_remap",O);guard=loadmod("refresh_generation002_for_bary_remap",G)
 face=c["source_O_faces"];detail=c["parent_fine_detail_face_row"];drows=sorted(set(map(int,detail[prep.EX])))
 edge_map=owner.edge_map(face);ring=set(prep.EX)
 for r in prep.EX:
  for u,v in zip(face[r],np.roll(face[r],-1)):
   for inc in edge_map[(min(int(u),int(v)),max(int(u),int(v)))]:
    q=int(inc[0])
    if int(detail[q]) in drows:ring.add(q)
 out,owner_report=owner.propose_patch_flips(c,drows,sorted(ring),max_flips=4,max_rounds=1,min_angle_gain_deg=.5,min_normal_dot=0.0)
 if owner_report["accepted_flip_count"]!=1:raise ValueError("deterministic replay did not yield one expected flip")
 op=owner_report["operations"][0];rows=sorted(map(int,op["face_rows"]))
 if rows!=[19818,19819] or op["new_diagonal"]!=[18488,20901]:raise ValueError(f"unexpected flip operation {op}")
 parent_union=sorted(set(map(int,op["immediate_parent_union"])))
 if parent_union!=[19444]:raise ValueError(f"unexpected immediate parent union {parent_union}")
 if not np.array_equal(out["points_world_m"],c["points_world_m"]) or out["points_world_m"].dtype!=np.float32:raise ValueError("fixed Float32 coordinates changed")
 if not np.array_equal(out["source_O_faces"],out["area_reference_037_source_O_faces"]):raise ValueError("matched-037 topology is not synchronized")
 old_bary=np.asarray(out["refinement_face_parent_barycentric"]).copy()
 ptri=parent["area_reference_037_points_world_m_f64"][parent["source_O_faces"][19444]].astype(np.float64)
 D=np.column_stack((ptri[0]-ptri[2],ptri[1]-ptri[2]))
 if np.linalg.matrix_rank(D)!=2:raise ValueError("immediate parent reference triangle is rank deficient")
 fresh={};old_err={};new_err={};minw={};maxsum={};rank_info={}
 for r in rows:
  q=out["area_reference_037_points_world_m_f64"][out["source_O_faces"][r]].astype(np.float64)
  uv,_,rank,_=np.linalg.lstsq(D,(q-ptri[2]).T,rcond=None)
  w=np.column_stack((uv.T,1.0-uv.sum(axis=0)))
  if rank!=2:raise ValueError(f"parent barycentric solve lost rank for row {r}")
  reconstructed=w@ptri
  fresh[r]=w;new_err[r]=float(np.max(np.abs(reconstructed-q)))
  old_err[r]=float(np.max(np.abs(old_bary[r]@ptri-q)))
  minw[r]=float(np.min(w));maxsum[r]=float(np.max(np.abs(w.sum(axis=1)-1.0)))
  rank_info[r]=rank
  out["refinement_face_parent_barycentric"][r]=w
  aptr=out["refinement_face_parent_ancestry_indptr"];av=out["refinement_face_parent_ancestry_face_rows"]
  prow=av[int(aptr[r]):int(aptr[r+1])]
  if prow.tolist()!=[19444] or int(out["refinement_face_parent_face_row"][r])!=19444:raise ValueError(f"parent CSR/scalar changed unexpectedly at {r}: {prow.tolist()}")
  if np.min(w)<-2e-8 or np.max(w)>1+2e-8 or maxsum[r]>2e-8 or new_err[r]>1e-12:raise ValueError(f"recomputed barycentric map fails at row {r}")
  if old_err[r]<1e-4:raise ValueError(f"expected stale owner map not observed at row {r}")
  if np.min(w)<-1e-8 or np.max(w)>1+1e-8:raise ValueError(f"corrected map too close to simplex bound at row {r}")
 # Only barycentric maps differ between raw owner proposal and corrected proposal.
 changed_rows=np.flatnonzero(np.any(out["refinement_face_parent_barycentric"]!=old_bary,axis=(1,2))).tolist()
 if changed_rows!=rows:raise ValueError(f"barycentric repair changed rows {changed_rows}, expected {rows}")
 owner.validate_candidate(out)
 guard_before=guard.validate_ancestry(c,parent,audit)
 guard_after=guard.validate_ancestry(out,parent,audit)
 # Explicit local Float32 shape check for the two new faces.
 shape=[];x=out["points_world_m"]
 for r in rows:
  tri=out["source_O_faces"][r];ar=prep.area(x,tri);ang=prep.amin(x,tri);asp=prep.asp(x,tri)
  shape.append({"row":r,"triangle":list(map(int,tri)),"area_m2":ar,"min_angle_deg":ang,"aspect":asp})
 if not all(q["area_m2"]>0 and q["min_angle_deg"]>=.5 and q["aspect"]<=50 for q in shape):raise ValueError(f"corrected output fails local F32 shape: {shape}")
 OUT.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(CP,**out)
 candidate_sha=sha(CP)
 if preexisting_sha is not None and preexisting_sha!=candidate_sha:raise ValueError("deterministic rerun output differs from previously written candidate")
 report={
  "schema":"numi.human.b43-16.parent-barycentric-remap.v1","status":"PRIVATE_CORRECTED_TOPOLOGY_CANDIDATE_PENDING_EXACT_LOCAL_AUDITS",
  "geometry_admitted":False,"native_admission":False,"accepted_physical_steps":0,
  "candidate_input_path":str(C),"candidate_input_sha256":PIN["candidate"],"immediate_input_path":str(P),"immediate_input_sha256":PIN["parent"],
  "candidate_output_path":str(CP),"candidate_output_sha256":candidate_sha,"preexisting_incomplete_candidate_sha256":preexisting_sha,"builder_script_path":str(Path(__file__).resolve()),"builder_script_sha256":sha(Path(__file__)),
  "inputs":{k:{"path":str(paths[k]),"sha256":PIN[k]} for k in paths},
  "preserved_failed_evidence_report_path":str(R),"preserved_failed_evidence_report_sha256":PIN["ring_report"],
  "operation":op,"changed_barycentric_rows_only":[int(x) for x in changed_rows],
  "reference_source":"immutable area_reference_037_points_world_m_f64 for emitted vertices and immediate parent 19444 triangle from pinned input candidate area_reference_037_points_world_m_f64; deformed points were not used",
  "parent_reference_face_row":19444,"parent_reference_triangle_m":ptri.tolist(),"recomputed_barycentric_rows":{str(k):v.tolist() for k,v in fresh.items()},
  "stored_owner_map_max_vertex_error_m":{str(k):v for k,v in old_err.items()},"corrected_map_max_vertex_error_m":{str(k):v for k,v in new_err.items()},
  "corrected_map_min_weight":{str(k):v for k,v in minw.items()},"corrected_map_row_sum_error":{str(k):v for k,v in maxsum.items()},
  "simplex_tolerance":2e-8,"reconstruction_tolerance_m":1e-12,"parent_rank":rank_info,
  "source_detail_rows":[8269],"fine_subface_rows":[8271],"source_detail_ancestry":[8269],"fine_subface_ancestry":[8271],"face_kind":2,"immediate_parent_union":[19444],
  "points_world_m_dtype":str(out["points_world_m"].dtype),"points_bitwise_unchanged":bool(np.array_equal(out["points_world_m"],c["points_world_m"])),"matched_037_topology_synchronized":bool(np.array_equal(out["source_O_faces"],out["area_reference_037_source_O_faces"])),
  "local_float32_shape":shape,"local_shape_gate":{"min_angle_deg":.5,"max_aspect":50,"positive_area":True,"passes":True},
  "generation002_ancestry_guard_before":guard_before,"generation002_ancestry_guard_after_barycentric_repair":guard_after,
  "generation002_guard_note":"The guard validates CSR/scalar/simplex semantics in legacy union mode; explicit per-face immutable-reference reconstruction for the two edited rows is separately measured above.",
  "source_material_transfer":"Same registered source detail/subface/kind and one immediate parent 19444; source identity remains representable. Continuous physical material, volume and layer transfer are not recomputed by the topology owner.",
  "exact_updated_bone_check":"NOT_RUN_PENDING_SHARED_GEOMETRY_SLOT",
  "exact_updated_self_check":"NOT_RUN_PENDING_SHARED_GEOMETRY_SLOT",
  "inside_node_check":"NOT_RUN_PENDING_SHARED_GEOMETRY_SLOT",
  "limitations":["candidate is construction-only and not admitted","the flip changes two rows that were baseline 43/16 exact crossing witnesses","updated exact bone, self and inside-node checks remain required","no physical transfer or admission"]}
 REPORT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False,default=lambda o:o.item() if isinstance(o,np.generic) else o.tolist() if isinstance(o,np.ndarray) else (_ for _ in ()).throw(TypeError(type(o).__name__)))+"\n")
 print(json.dumps({"candidate_path":str(CP),"candidate_sha256":sha(CP),"report_path":str(REPORT),"report_sha256":sha(REPORT),"changed_bary_rows":changed_rows,"old_errors_m":old_err,"new_errors_m":new_err,"min_weights":minw,"generation002_after_contract":guard_after["refinement_parent_contract_mode"],"shape":shape,"exact_audits":"NOT_RUN_PENDING_SHARED_GEOMETRY_SLOT"},indent=2,sort_keys=True))
if __name__=="__main__":main()
