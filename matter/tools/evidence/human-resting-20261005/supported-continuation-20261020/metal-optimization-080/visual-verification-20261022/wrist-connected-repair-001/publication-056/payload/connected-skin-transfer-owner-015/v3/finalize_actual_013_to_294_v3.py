#!/usr/bin/env python3
import hashlib, importlib.util, json, pathlib, struct, sys
import numpy as np
R=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=R/"E/connected-skin-transfer-owner-015"; inp=D/"actual-013-to-294-input.bin"; out=D/"actual-013-to-294-two-way-map-v3.bin"; prep=json.loads((D/"actual-013-to-294-input.json").read_text())
npzout=D/"actual-013-to-294-two-way-embedding-v3.npz"; reportpath=D/"actual-013-to-294-two-way-embedding-v3.json"
if npzout.exists() or reportpath.exists(): raise SystemExit("refusing overwrite")
if hashlib.sha256(inp.read_bytes()).hexdigest()!=prep["input_binary"]["sha256"]: raise SystemExit("prepared input hash changed")
with inp.open("rb") as f:
 ns,nf,no,nof=struct.unpack("<4Q",f.read(32)); src=np.frombuffer(f.read(ns*3*4),dtype="<f4").reshape(ns,3).copy(); sf=np.frombuffer(f.read(nf*3*4),dtype="<u4").reshape(nf,3).copy(); weights=np.frombuffer(f.read(ns*86*4),dtype="<f4").reshape(ns,86).copy(); bodies=np.frombuffer(f.read(86*4),dtype="<u4").copy(); outer=np.frombuffer(f.read(no*3*4),dtype="<f4").reshape(no,3).copy(); of=np.frombuffer(f.read(nof*3*4),dtype="<u4").reshape(nof,3).copy()
with out.open("rb") as f:
 no2,ninferred,nexplicit,nactive,nmapped,ninactive,nfallback=struct.unpack("<7Q",f.read(56)); row=np.frombuffer(f.read(no2*3*4),dtype="<f4").reshape(no2,3).copy(); bary=np.frombuffer(f.read(no2*3*8),dtype="<f8").reshape(no2,3).copy(); residual=np.frombuffer(f.read(no2*3*8),dtype="<f8").reshape(no2,3).copy(); o_support=np.frombuffer(f.read(no2*3*4),dtype="<u4").reshape(no2,3).copy(); donor=np.frombuffer(f.read(no2*4),dtype="<u4").copy(); active=np.frombuffer(f.read(ns),dtype=np.uint8).copy(); mapped=np.frombuffer(f.read(ns),dtype=np.uint8).copy(); d_support=np.frombuffer(f.read(ns*3*4),dtype="<u4").reshape(ns,3).copy(); d_weights=np.frombuffer(f.read(ns*3*4),dtype="<f4").reshape(ns,3).copy(); d_distance=np.frombuffer(f.read(ns*4),dtype="<f4").copy(); fallback_face=np.frombuffer(f.read(nfallback*4*4),dtype="<u4").reshape(nfallback,4).copy(); fallback_support=np.frombuffer(f.read(nfallback*3*4),dtype="<u4").reshape(nfallback,3).copy(); fallback_weights=np.frombuffer(f.read(nfallback*3*4),dtype="<f4").reshape(nfallback,3).copy(); fallback_distance=np.frombuffer(f.read(nfallback*4),dtype="<f4").copy()
 if f.read(1): raise SystemExit("unexpected trailing v3 map bytes")
face_rows=row[:,0].astype(np.uint32); projection=row[:,1].astype(np.float64)
if (no2,ninferred,nexplicit,nactive,ninactive,nfallback)!=(no,no,0,int(active.sum()),ns-int(active.sum()),26) or nmapped!=int(mapped.sum())+nfallback: raise SystemExit(f"v3 map header/count mismatch {(no2,ninferred,nexplicit,nactive,nmapped,ninactive,nfallback)}")
if np.any(face_rows>=nf) or not np.isfinite(bary).all() or np.any(bary < -1e-9) or np.any(bary > 1+1e-9): raise SystemExit("invalid source face/barycentric map")
if np.max(np.abs(bary.sum(axis=1)-1.0))>2e-12: raise SystemExit("source barycentric rows do not close")
closure=np.linalg.norm(np.einsum("ni,nij->nj",bary,src[sf[face_rows]])+residual-outer,axis=1)
if float(closure.max())>1e-11: raise SystemExit("source residual does not reconstruct O")
interp=np.einsum("ni,nij->nj",bary,weights[sf[face_rows]])
expected=bodies[np.argmax(interp,axis=1)]
if np.count_nonzero(expected!=donor): raise SystemExit("O donor differs from barycentric 86-lane argmax")
vertex_donor=bodies[np.argmax(weights,axis=1)]
for i in range(no):
 if donor[i] not in vertex_donor[sf[face_rows[i]]]: raise SystemExit(f"assigned O donor absent from projected source face {i}")
active_mask=active.astype(bool); mapped_mask=mapped.astype(bool)
source_face_used=np.zeros(ns,dtype=bool); source_face_used[sf.reshape(-1)]=True
if not np.array_equal(active_mask,source_face_used): raise SystemExit("active detail mask is not exactly registered-face-referenced source vertices")
if np.any(mapped_mask & ~active_mask): raise SystemExit("unreferenced detail vertex has a transfer row")
if int(active_mask.sum())!=nactive or int(mapped_mask.sum())+nfallback!=nmapped or int((~active_mask).sum())!=ninactive: raise SystemExit("detail active/mapped/inactive header counts differ")
physical_donor_set=set(map(int,donor))
expected_mapped=active_mask & np.isin(vertex_donor,np.fromiter(physical_donor_set,dtype=np.uint32))
if not np.array_equal(mapped_mask,expected_mapped): raise SystemExit("mapped detail rows differ from face-used rows whose dominant donors occur on physical O")
direct_mapped_mask=mapped_mask.copy()
unmapped_mask=active_mask & ~direct_mapped_mask
if not np.isfinite(d_weights[direct_mapped_mask]).all() or not np.isfinite(d_distance[direct_mapped_mask]).all(): raise SystemExit("nonfinite mapped detail-to-O rows")
if direct_mapped_mask.any() and np.max(np.abs(d_weights[direct_mapped_mask].sum(axis=1)-1.0))>2e-5: raise SystemExit("detail-to-O transfer rows do not normalize")
if np.any(d_support[direct_mapped_mask]>=no) or np.any(d_weights[direct_mapped_mask]<-1e-7): raise SystemExit("invalid detail-to-O supports")
outer_face_keys={tuple(sorted(map(int,face))) for face in of}
for i in np.flatnonzero(direct_mapped_mask):
 owner=int(vertex_donor[i]); support=d_support[i]
 if tuple(sorted(map(int,support))) not in outer_face_keys: raise SystemExit(f"detail-to-O support is not an emitted physical O face {i}")
 if owner not in set(map(int,donor[support])): raise SystemExit(f"detail donor absent from selected physical O face {i}")
if np.any(~direct_mapped_mask) and (np.any(d_weights[~direct_mapped_mask]!=0) or np.any(d_support[~direct_mapped_mask]!=0) or np.any(d_distance[~direct_mapped_mask]!=0)): raise SystemExit("direct-unmapped detail row contains fabricated sparse mapping")
rep_report_path=D/"representative-source-graph-diagnostic-v2.json"
rep_report=json.loads(rep_report_path.read_text())
rep_rows=rep_report["rows"]
if len(rep_rows)!=nfallback or rep_report["source_payload_sha256"]!=prep["source_294"]["sha256"]: raise SystemExit("source-graph representative report does not match v3 source rows")
fallback_source=np.asarray(fallback_face[:,0],dtype=np.uint32)
fallback_rep=np.asarray(fallback_face[:,1],dtype=np.uint32)
fallback_face_index=np.asarray(fallback_face[:,2],dtype=np.uint32)
fallback_donor=np.asarray(fallback_face[:,3],dtype=np.uint32)
expected_fallback=np.flatnonzero(active_mask & ~direct_mapped_mask).astype(np.uint32)
if not np.array_equal(fallback_source,expected_fallback): raise SystemExit("fallback rows do not exactly cover the active donor-unsupported source rows")
rep_by_source={int(r["source_vertex"]):r for r in rep_rows}
if set(map(int,fallback_source))!=set(rep_by_source): raise SystemExit("graph representative list differs from C++ fallback rows")
xmedian=float(np.median(src[:,0]))
graph_edge_set=set()
for face in sf:
 for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
  graph_edge_set.add((min(int(a),int(b)),max(int(a),int(b))))
body_pose=json.loads(pathlib.Path(prep["snapshot"]["path"]).read_text())
pose_bodies={int(q["body_index"]):q for q in body_pose.get("all_bodies",body_pose.get("bodies",[]))}
rep_rows_by_source=np.full(ns,np.uint32(0xffffffff),dtype=np.uint32)
rep_face_by_source=np.full(ns,np.uint32(0xffffffff),dtype=np.uint32)
fallback_mask=np.zeros(ns,dtype=bool)
fallback_owner_distance=[]
for j,srcid0 in enumerate(fallback_source):
 srcid=int(srcid0); rep=int(fallback_rep[j]); fi=int(fallback_face_index[j]); own=int(fallback_donor[j]); rr=rep_by_source[srcid]
 if rep!=int(rr["representative_o_local"]) or own!=int(vertex_donor[srcid]): raise SystemExit(f"fallback representative/donor differs from graph lineage row {srcid}")
 if fi>=len(of) or rep>=no: raise SystemExit(f"fallback representative or face is out of range row {srcid}")
 if rep not in set(map(int,of[fi])): raise SystemExit(f"fallback selected face is outside representative one-ring row {srcid}")
 if tuple(sorted(map(int,fallback_support[j])))!=tuple(sorted(map(int,of[fi]))): raise SystemExit(f"fallback support differs from selected physical O face row {srcid}")
 if not np.isfinite(fallback_weights[j]).all() or np.any(fallback_weights[j]<0) or abs(float(fallback_weights[j].sum())-1.0)>1e-4: raise SystemExit(f"invalid fallback barycentric weights row {srcid}")
 dist=float(fallback_distance[j]); parent=float(np.linalg.norm(src[srcid].astype(float)-outer[rep].astype(float)))
 if not np.isfinite(dist) or dist<0 or dist>0.05 or dist>parent+1e-9: raise SystemExit(f"fallback representative one-ring projection cap failed row {srcid}")
 path=list(map(int,rr["source_graph_path_vertex_ids"]))
 if not path or path[0]!=srcid or path[-1]!=int(rr["anchor_vertex"]): raise SystemExit(f"source graph path endpoints invalid row {srcid}")
 if rr["source_component"]!=rr["anchor_component"]: raise SystemExit(f"source graph anchor leaves connected component row {srcid}")
 for a,b in zip(path[:-1],path[1:]):
  if (min(a,b),max(a,b)) not in graph_edge_set: raise SystemExit(f"source graph witness uses a non-edge row {srcid}")
 sfrow=int(rr["anchor_source_face_row"]); anchor=int(rr["anchor_vertex"])
 if sfrow>=nf or anchor not in set(map(int,sf[sfrow])) or sfrow!=int(face_rows[rep]): raise SystemExit(f"graph/source-face crosswalk invalid row {srcid}")
 if np.max(np.abs(np.asarray(rr["anchor_barycentric"])-bary[rep]))>1e-12: raise SystemExit(f"graph anchor barycentrics differ from O->source embedding row {srcid}")
 if int(vertex_donor[srcid])==102:
  if min(src[path,0])<=xmedian or outer[rep,0]<=xmedian: raise SystemExit(f"right-side source region crossed centerline row {srcid}")
 elif int(vertex_donor[srcid])==10:
  center=pose_bodies.get(10,{}).get("current",{}).get("position_world_m")
  if center is None or abs(float(center[0])-xmedian)>=0.01: raise SystemExit("body 10 cannot be classified as side-neutral from pinned pose")
 else: raise SystemExit(f"unexpected unsupported source donor {int(vertex_donor[srcid])}")
 d_support[srcid]=fallback_support[j]; d_weights[srcid]=fallback_weights[j]; d_distance[srcid]=fallback_distance[j]
 fallback_mask[srcid]=True; rep_rows_by_source[srcid]=rep; rep_face_by_source[srcid]=fi; fallback_owner_distance.append(dist)
mapped_mask=direct_mapped_mask|fallback_mask
unmapped_mask=active_mask&~mapped_mask
if int(fallback_mask.sum())!=nfallback or not np.array_equal(mapped_mask,active_mask): raise SystemExit("v3 fallback did not map the complete active source domain")
if np.any(~mapped_mask) and (np.any(d_weights[~mapped_mask]!=0) or np.any(d_support[~mapped_mask]!=0) or np.any(d_distance[~mapped_mask]!=0)): raise SystemExit("inactive detail rows contain fabricated sparse mapping")

npz=R/"E/connected-skin-mechanical-transfer-013/boundary-refinement-002/candidate.npz"
with np.load(npz,allow_pickle=False) as z:
 global_o=np.asarray(z["physical_outer_node_ids"],dtype="<u4"); points=np.asarray(z["points_world_m"],dtype="<f4"); iffaces=np.asarray(z["physical_outer_faces"],dtype="<i8")
 rev=np.full(len(points),-1,dtype=np.int64);rev[global_o]=np.arange(len(global_o));iflocal=rev[iffaces].astype("<u4")
 if not np.array_equal(points[global_o],outer) or not np.array_equal(iflocal,of): raise SystemExit("candidate O topology/positions differ from transfer inputs")
patch_path=R/"global-skin-crossing-patches-012/connected-physical-O-patches-012.npz"
with np.load(patch_path,allow_pickle=False) as z:
 patch_node_ids=np.asarray(z["patch_node_ids"],dtype=np.int64)
 patch_node_points=np.asarray(z["patch_node_points_world_m"],dtype="<f4")
 patch_offsets=np.asarray(z["patch_node_offsets"],dtype=np.int64)
 o_row_by_global={int(v):i for i,v in enumerate(global_o)}
 if len(o_row_by_global)!=len(global_o): raise SystemExit("physical O global node IDs are not unique")
 patch_to_o=np.asarray([o_row_by_global.get(int(v),-1) for v in patch_node_ids],dtype=np.int64)
 if np.any(patch_to_o<0) or len(patch_to_o)!=len(patch_node_ids): raise SystemExit("012 patch node missing from 013 physical O rows")
 if not np.array_equal(patch_node_points,outer[patch_to_o]): raise SystemExit("012 patch coordinates differ from 013 physical O rows")
 if patch_offsets.ndim!=1 or patch_offsets[0]!=0 or patch_offsets[-1]!=len(patch_to_o) or np.any(np.diff(patch_offsets)<0): raise SystemExit("invalid 012 patch node offsets")
reason=np.full(ns,2,dtype=np.uint8); reason[mapped_mask]=0; reason[unmapped_mask]=1
# 0=mapped directly or through source-connected representative, 1=active donor unsupported and unresolved, 2=not face-referenced.
patch_reports=[]
for pi,(lo,hi) in enumerate(zip(patch_offsets[:-1],patch_offsets[1:])):
 rr=patch_to_o[lo:hi]; body_rows=[]
 for bid in np.unique(donor[rr]):
  sel=rr[donor[rr]==bid]; ds=projection[sel]
  body_rows.append({"binding_body_index":int(bid),"rows":int(sel.size),"max_projection_m":float(ds.max()),"rms_projection_m":float(np.sqrt(np.mean(ds.astype(float)**2)))})
 patch_reports.append({"patch_index":pi,"node_rows":int(hi-lo),"donor_rows":body_rows})
o_by_donor={}
for bid in np.unique(donor):
 sel=donor==bid; ds=projection[sel]
 o_by_donor[str(int(bid))]={"rows":int(sel.sum()),"max_projection_m":float(ds.max()),"rms_projection_m":float(np.sqrt(np.mean(ds.astype(float)**2))),"max_residual_m":float(np.linalg.norm(residual[sel],axis=1).max())}
detail_by_donor={}
for bid in np.unique(vertex_donor[active_mask]):
 sel=active_mask&(vertex_donor==bid); ok=mapped_mask&sel; bad=sel&~mapped_mask
 detail_by_donor[str(int(bid))]={"face_referenced_rows":int(sel.sum()),"mapped_rows":int(ok.sum()),"unmapped_rows":int(bad.sum()),"unmapped_reason":"dominant source body has no physical-O donor faces" if bad.any() else None,"max_projection_m":float(d_distance[ok].max()) if ok.any() else None,"rms_projection_m":float(np.sqrt(np.mean(d_distance[ok].astype(float)**2))) if ok.any() else None}
np.savez_compressed(npzout,registered_current_world_m=src,registered_faces=sf,registered_weights86=weights,binding_body_indices=bodies,physical_outer_global_volume_node_ids=global_o,physical_outer_world_m=outer,physical_outer_faces_local=of,o_to_detail_source_face_rows=face_rows,o_to_detail_source_barycentric=bary,o_to_detail_source_residual_world_m=residual,o_to_detail_dominant_body_indices=donor,o_to_detail_projection_distance_m=projection,o_to_detail_support_source_vertex_ids=o_support,detail_to_o_active=active,detail_to_o_mapped=mapped_mask.astype(np.uint8),detail_to_o_unmapped_reason_code=reason,detail_to_o_dominant_body_indices=vertex_donor,detail_to_o_support_outer_rows=d_support,detail_to_o_weights=d_weights,detail_to_o_projection_distance_m=d_distance,detail_to_o_source_connected_fallback=fallback_mask,detail_to_o_representative_coarse_local_row=rep_rows_by_source,detail_to_o_representative_selected_face_row=rep_face_by_source,patch012_node_ids=patch_node_ids,patch012_node_to_o_local_row=patch_to_o,patch012_node_points_world_m=patch_node_points,patch012_node_offsets=patch_offsets)
snapshot4=pathlib.Path(prep["snapshot"]["path"]); snapshot7=R/"native-app-main-header-integration-151/source-preflight-007/registered-skin-poses.json"
a4=json.loads(snapshot4.read_text());a7=json.loads(snapshot7.read_text())
def bodies_by_id(j): return {int(x["body_index"]):x for x in j.get("all_bodies",j.get("bodies",[]))}
x4,x7=bodies_by_id(a4),bodies_by_id(a7); shared=sorted(set(x4)&set(x7)); pose007004_exact=(len(shared)==86 and all(x4[i]["current"]==x7[i]["current"] for i in shared))
source=pathlib.Path(prep["source_294"]["path"]); helper=R/"regional-refinement-compose-255/tools/rebind_human_skin_reference.py"
ownerfiles=[D/"run_actual_013_to_294_v3.cpp",D/"ConnectedSkinTransferOwner015.hpp",R/"wholebody-attachment-source-rebind-033-final091-regenerated-deltas-001/source/apps/NumiHumanAcceptedFEMSkinTransfer.hpp",helper]
sourceface_usage=np.bincount(face_rows,minlength=nf)
rec={"schema":"numi.human.connected-skin-transfer-015.actual-013-to-294-two-way-inferred.v1","pins":{"candidate_013_sha256":prep["candidate_013"]["sha256"],"source_294_payload_sha256":prep["source_294"]["sha256"],"snapshot_004_sha256":prep["snapshot"]["sha256"],"snapshot_007_sha256":hashlib.sha256(snapshot7.read_bytes()).hexdigest(),"accepted_transfer_owner_path":str(ownerfiles[2]),"accepted_transfer_owner_sha256":hashlib.sha256(ownerfiles[2].read_bytes()).hexdigest(),"projection_owner_sha256":hashlib.sha256(ownerfiles[1].read_bytes()).hexdigest(),"runner_sha256":hashlib.sha256(ownerfiles[0].read_bytes()).hexdigest(),"representative_map_binary_sha256":hashlib.sha256((D/"representative-source-graph-v2.bin").read_bytes()).hexdigest(),"representative_graph_diagnostic_sha256":hashlib.sha256(rep_report_path.read_bytes()).hexdigest(),"representative_graph_builder_sha256":hashlib.sha256((D/"source-connected-representative-diagnostic-v2.py").read_bytes()).hexdigest(),"replay_helper_sha256":hashlib.sha256(helper.read_bytes()).hexdigest(),"input_binary_sha256":prep["input_binary"]["sha256"],"output_binary_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"patch_012_npz_sha256":hashlib.sha256(patch_path.read_bytes()).hexdigest()},"frame":{"source":"source294 ABI5 full-weight vertices replayed with published exact 255 helper from identity-004 current body transforms","physical_O":"013-v2 physical O rows in actual007 coordinate frame","identity_007_004_current_body_poses_exact_for_shared_86_bindings":pose007004_exact,"source_294_binding_records_sha256":"06b78f0b9ef114193c81caac7373b9fc8dd8e68374c627cec5bbe5c31c0a09c2","native_registered_skin_csv_used_for_coordinates_or_donors":False,"native_csv_caveat":{"root_independent_measurement_max_coordinate_delta_vs_replayed_294_m":5.98463e-7,"root_independent_measurement_csv_donor_vs_294_full_weight_argmax_mismatch_rows":11512}},"method":{"o_to_detail":"Existing AcceptedFEMSkinTransfer two-pass build: unrestricted donor probe; donor resolver uses first argmax of exact barycentric 86-lane source weights and payload binding bodyIndex; second build uses donor-filtered registered 294 triangle BVHs. Per-row dense source face ordinal, barycentrics, explicit O minus source-point residual and projection distance retained.","detail_to_o":"Existing AcceptedFEMSkinTransfer owner maps donor-supported rows directly. The 26 active rows whose dominant donors have no physical-O faces use its existing detailRepresentativeCoarseIndex path with sourceConnectedFaceRings=1. Representatives are selected by deterministic shortest path on the exact registered-294 face graph from the unsupported source vertex to the source-face anchor of an O row; original detail donor IDs remain unchanged. Owner searches only the representative physical-O one-ring and enforces projection <= representative Euclidean distance and the 0.05m projection cap. Representative rows are inferred construction support, not measured ancestry.","provenance":"INFERRED_CONSTRUCTION_EMBEDDING; geometric projection in matched current world frame, not measured ancestry","projection_bound_m":0.05,"source_connected_fallback_rows":nfallback,"fallback_projection_max_m":float(max(fallback_owner_distance)),"fallback_projection_rms_m":float(np.sqrt(np.mean(np.square(fallback_owner_distance)))),"fallback_graph_path_max_m":float(max(r["graph_plus_anchor_connector_m"] for r in rep_rows)),"fallback_graph_path_cap_applies":False,"laterality_gate":"body 102 requires same positive-x side along its source graph path and at representative; body 10 is side-neutral by pinned current body-center proximity to source x median"},"counts":{"o_rows":no,"o_to_detail_mapped":int(ninferred),"o_rows_with_measured_source_ancestry":int(nexplicit),"source_faces":nf,"source_faces_used":int(np.count_nonzero(sourceface_usage)),"detail_source_face_referenced_rows":int(active_mask.sum()),"detail_source_unreferenced_rows":int((~active_mask).sum()),"detail_to_o_mapped_rows":int(mapped_mask.sum()),"detail_to_o_active_unmapped_rows":int(unmapped_mask.sum()),"detail_to_o_unmapped_donor_ids":sorted(map(int,np.unique(vertex_donor[unmapped_mask]))),"distinct_o_donor_body_ids":int(len(np.unique(donor))),"o_to_detail_by_donor":o_by_donor,"detail_to_o_by_donor":detail_by_donor,"source_connected_fallback_by_donor":{str(int(b)):int(np.count_nonzero(fallback_mask & (vertex_donor==b))) for b in np.unique(vertex_donor[fallback_mask])},"source_connected_fallback_representative_rows":int(fallback_mask.sum()),"patch012_node_rows":int(len(patch_to_o)),"patch012_patches":patch_reports},"metrics":{"o_to_detail_max_projection_m":float(projection.max()),"o_to_detail_rms_projection_m":float(np.sqrt(np.mean(projection**2))),"o_to_detail_max_residual_closure_m":float(closure.max()),"o_to_detail_max_barycentric_sum_error":float(np.max(np.abs(bary.sum(axis=1)-1))),"o_to_detail_dominant_donor_mismatches":int(np.count_nonzero(expected!=donor)),"detail_to_o_max_projection_m":float(d_distance[mapped_mask].max()),"detail_to_o_rms_projection_m":float(np.sqrt(np.mean(d_distance[mapped_mask].astype(float)**2))),"detail_to_o_transfer_row_max_partition_error":float(np.max(np.abs(d_weights[mapped_mask].sum(axis=1)-1))),"source_connected_fallback_projection_max_m":float(max(fallback_owner_distance)),"source_connected_fallback_projection_rms_m":float(np.sqrt(np.mean(np.square(fallback_owner_distance)))) ,"source_connected_fallback_graph_path_max_m":float(max(r["graph_plus_anchor_connector_m"] for r in rep_rows))},"artifact":{"path":str(npzout),"sha256":hashlib.sha256(npzout.read_bytes()).hexdigest(),"format":"NPZ arrays accompanying existing source/physical surface; no new runtime payload format"},"status":"all 54,551 face-referenced detail rows have a transfer row. Twenty-six unsupported-donor rows use the existing source-connected representative fallback; this is an inferred construction map, not anatomical ownership proof. The 50 mm owner projection cap is a construction bound, not anatomical acceptance. Geometry, mass, pose, and native admission remain closed","physical_or_native_admission":False}
reportpath.write_text(json.dumps(rec,indent=2)+"\n")
print(json.dumps({"report":str(reportpath),"report_sha256":hashlib.sha256(reportpath.read_bytes()).hexdigest(),"npz":str(npzout),"npz_sha256":rec["artifact"]["sha256"],"rows":{"o":no,"detail_face_referenced":int(active_mask.sum()),"detail_mapped":int(mapped_mask.sum()),"detail_active_unmapped":int(unmapped_mask.sum()),"detail_unreferenced":int((~active_mask).sum())},"o_to_detail":{"max_m":rec["metrics"]["o_to_detail_max_projection_m"],"rms_m":rec["metrics"]["o_to_detail_rms_projection_m"],"closure_m":rec["metrics"]["o_to_detail_max_residual_closure_m"]},"detail_to_o":{"max_m":rec["metrics"]["detail_to_o_max_projection_m"],"rms_m":rec["metrics"]["detail_to_o_rms_projection_m"]}},indent=2))
