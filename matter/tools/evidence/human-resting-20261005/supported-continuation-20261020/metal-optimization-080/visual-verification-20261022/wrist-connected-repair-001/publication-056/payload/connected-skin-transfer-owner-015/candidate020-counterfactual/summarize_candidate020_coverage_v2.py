import hashlib,importlib.util,json,pathlib,sys
import numpy as np
R=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
C=R/"E/connected-skin-transfer-owner-015"; Q=C/"candidate020-counterfactual"
base=C/"actual-013-to-294-two-way-embedding-v3.npz"
cand=Q/"actual-013-to-294-020-two-way-embedding-v2.npz"
selectedpath=R/"bilateral-hand-functional-rebind-020/run-001/selected-hand-rows.npz"
basej=json.loads((C/"actual-013-to-294-two-way-embedding-v3.json").read_text())
candj=json.loads((Q/"actual-013-to-294-020-two-way-embedding-v2.json").read_text())
patch=json.loads((R/"bilateral-hand-functional-rebind-020/run-001/report.json").read_text())
with np.load(selectedpath,allow_pickle=False) as z:
 right_ids=z["right_source_vertex_ids"].copy(); left_ids=z["left_source_vertex_ids"].copy()
with np.load(base,allow_pickle=False) as a,np.load(cand,allow_pickle=False) as b:
 x0=a["registered_current_world_m"].astype(np.float32);x1=b["registered_current_world_m"].astype(np.float32)
 f0=a["registered_faces"];f1=b["registered_faces"]
 w0=a["registered_weights86"];w1=b["registered_weights86"]
 bind=a["binding_body_indices"].astype(int)
 active=b["detail_to_o_active"].astype(bool);mapped=b["detail_to_o_mapped"].astype(bool);fb=b["detail_to_o_source_connected_fallback"].astype(bool)
 donor=bind[np.argmax(w1,axis=1)]
 old=bind[np.argmax(w0,axis=1)]
 hand=np.unique(np.r_[patch["coverage"]["right"]["after_dominant_body_ids"],patch["coverage"]["left"]["after_dominant_body_ids"]]).astype(int)
 oDon=np.unique(b["o_to_detail_dominant_body_indices"].astype(int))
 def rows_for(ids):
  ids=np.asarray(ids,dtype=int); m=np.zeros(len(donor),bool);m[ids]=True;m &= active
  return {"source_rows":int(m.sum()),"mapped_rows":int(np.count_nonzero(mapped&m)),"direct_rows":int(np.count_nonzero(mapped&m&~fb)),"fallback_rows":int(np.count_nonzero(fb&m)),"unmapped_active_rows":int(np.count_nonzero(m&~mapped)),"dominant_lane_rows":{str(int(k)):int(np.count_nonzero(m&(donor==k))) for k in np.unique(donor[m])}}
 from collections import Counter
 trans=Counter((int(old[i]),int(donor[i])) for i in np.flatnonzero(active) if old[i]!=donor[i])
 d=np.linalg.norm(x1.astype(float)-x0.astype(float),axis=1)
 hand_rows=np.isin(donor,hand)&active
 out={
 "schema":"numi.human.connected-skin-transfer-015.candidate020-coarse-owner-coverage.v2",
 "pins":{"candidate020_nhskin_sha256":"7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb","candidate020_selected_hand_rows_sha256":hashlib.sha256(selectedpath.read_bytes()).hexdigest(),"candidate020_input_sha256":hashlib.sha256((Q/"actual-013-to-294-020-input.bin").read_bytes()).hexdigest(),"candidate020_mapping_npz_sha256":hashlib.sha256(cand.read_bytes()).hexdigest(),"candidate020_mapping_report_sha256":hashlib.sha256((Q/"actual-013-to-294-020-two-way-embedding-v2.json").read_bytes()).hexdigest(),"baseline294_mapping_npz_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),"baseline294_mapping_report_sha256":hashlib.sha256((C/"actual-013-to-294-two-way-embedding-v3.json").read_bytes()).hexdigest(),"identity004_snapshot_sha256":"0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd","physical013_sha256":"1849d24c14f8c0ddfedf3aa131877bdd38a852f9e48baebcf900a0efadf75db1"},
 "input_consistency":{"source_vertex_count":int(len(x1)),"source_face_count":int(len(f1)),"same_source_face_indices_as_294":bool(np.array_equal(f0,f1)),"face_referenced_active_rows":int(active.sum()),"all_active_rows_mapped":bool(np.array_equal(mapped,active)),"active_unmapped_rows":int(np.count_nonzero(active&~mapped)),"unreferenced_rows":int(np.count_nonzero(~active)),"current_position_changed_rows_vs_original294":int(np.count_nonzero(np.any(x0!=x1,axis=1))),"max_current_position_delta_vs_original294_m":float(d.max()),"rms_current_position_delta_vs_original294_m":float(np.sqrt(np.mean(d*d))),"weights_changed_rows":int(np.count_nonzero(np.any(w0!=w1,axis=1)))},
 "dominant_owner":{"active_rows_with_changed_argmax":int(sum(trans.values())),"transitions":[{"from_body_index":a,"to_body_index":b,"active_rows":n} for (a,b),n in sorted(trans.items())],"candidate020_hand_right":{"source_rows":int(patch["coverage"]["right"]["rows"]),"new_dominant_body_ids":patch["coverage"]["right"]["after_dominant_body_ids"],"source_row_coverage":rows_for(right_ids)},"candidate020_hand_left":{"source_rows":int(patch["coverage"]["left"]["rows"]),"new_dominant_body_ids":patch["coverage"]["left"]["after_dominant_body_ids"],"source_row_coverage":rows_for(left_ids)},"all_candidate_hand_dominant_body_ids":hand.tolist(),"hand_ids_present_on_physical_O_owner_map":sorted(set(hand.tolist())&set(oDon.tolist())),"hand_ids_missing_from_physical_O_owner_map":sorted(set(hand.tolist())-set(oDon.tolist())),"distinct_physical_O_dominant_body_ids_baseline294":basej["counts"]["distinct_o_donor_body_ids"],"distinct_physical_O_dominant_body_ids_candidate020":candj["counts"]["distinct_o_donor_body_ids"]},
 "transfer":{"active_rows":int(active.sum()),"mapped_rows":int(mapped.sum()),"direct_rows":int(np.count_nonzero(mapped&~fb)),"source_connected_fallback_rows":int(fb.sum()),"fallback_by_original_dominant_donor":candj["counts"]["source_connected_fallback_by_donor"],"fallback_representative_physical_O_donors":{str(int(q)):int(np.count_nonzero(fb & (b["detail_to_o_representative_coarse_local_row"]>=0) & np.isin(np.arange(len(donor)),np.arange(len(donor))))) for q in []},"max_owner_projection_m":candj["metrics"]["detail_to_o_max_projection_m"],"rms_owner_projection_m":candj["metrics"]["detail_to_o_rms_projection_m"],"configured_projection_cap_m":0.05,"cap_is_anatomical_acceptance":False},
 "admission":{"status":"counterfactual owner mapping only","source_embeddings":"INFERRED_CONSTRUCTION_EMBEDDING; not measured ancestry","physical_or_native_admission":False,"geometry_mass_pose_acceptance":False}
 }
 # Add actual O donor used for each source-connected fallback.
 rep=b["detail_to_o_representative_coarse_local_row"].astype(int); olabels=b["o_to_detail_dominant_body_indices"].astype(int)
 out["transfer"]["fallback_representative_physical_O_donors"]={str(int(k)):int(np.count_nonzero(fb & (olabels[rep.clip(0,len(olabels)-1)]==k))) for k in np.unique(olabels[rep[fb]])}
 p=Q/"candidate020-owner-coverage-v2.json"
 if p.exists(): raise SystemExit("refusing overwrite")
 p.write_text(json.dumps(out,indent=2)+"\n")
 print(json.dumps({"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"input_consistency":out["input_consistency"],"dominant_owner":out["dominant_owner"],"transfer":out["transfer"],"admission":out["admission"]},indent=2))
