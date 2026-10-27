#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,importlib.util,json,subprocess,sys,time
import numpy as np
BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/candidate.npz")
C=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b65-99/attempt-019-exact-witness-layer-guard-001/fit-run-008-alpha-0p25-corrected/candidate-private-alpha-0p25.npz")
R=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b65-99/attempt-019-exact-witness-layer-guard-001/fit-run-008-alpha-0p25-corrected/layer-identity-diff.json")
L=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/physical-layer-diagnostic-001/guarded-65-quarter/skin-fat-layers-541.npz")
AUDIT=BASE.parent/"audit-159/report.json"; SELF=BASE.parent/"audit-130/summary.json"; SELFP=BASE.parent/"audit-130/surface-self-pairs.npy"
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
AUD159=ROOT/"D/publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py"
AUD130=ROOT/"D/publication-181-stage-001/sources/checks/audit_connected_outer_geometry_130.py"
EXACT=E/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
BANKHELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
EXPORT=E/"E/generic-static-query-042/prepare_generic_query_points.py"
QUERYBIN=Path("/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query")
OUT=Path(__file__).resolve().parent/"scoped-exact-run-001"
EXPECTED={
"base":"a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db",
"candidate":"e51ff27acd6a29eefcb30783085c3f8af3b36b12ffd55d43ee714f1b872b380a",
"candidate_report":"7cee89a174863bd79a6d4f625cf15a1211be1ff6cc0f7bb661406a9295525b86",
"layer":"78916a0c05eb130e5efb02174342295c2d8d4444eae963c7b36fc43a2a1b17fc",
"audit159":"2e870a160a1514e7a5621ceb504845303e984871532f60278bac12b6bbf817a8",
"self_summary":"c1aa0b43b19cfa860ee7c92400ecd396d16f5442c0918bdb2ef4c434bfff6a0d",
"self_pairs":"55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de",
"audit159_source":"de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1",
"audit130_source":"52d45f287162d22dca417793c1393bcfbbe506c5ce85b828f009b4fd7ccd94b9",
"exact_predicate":"f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d",
"bank_helper":"ea833990d8b1ffb309d4960e62fb8c67e5cf0ecdbcdc0d8e9ad40c20eaf4c6db",
"query_exporter":"0e09fecb9cf0d2c5e9049356b822d138819e1560a2a2049bcbd4d8602336a1ee",
"query_binary":"5dc7b5ce2a96f58bd03ccc85e38645d9da2ec8be105a532c461a3e0cc6d25460"}
CHANGED=np.empty(0,dtype=np.int64)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadnpz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p)
 if sp is None or sp.loader is None:raise RuntimeError(f"cannot load {p}")
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def audit_changed_faces(world,faces,body,bone,T,ids,pos,M,offset,exact,aud):
 local=(world-pos[body])@M[body].T-offset[body]
 skin=local[faces[CHANGED]]
 mask=(ids[:,0]==body)&(ids[:,1]==bone)
 bank=T[mask];bankids=ids[mask,3]
 if not len(bank) or not len(skin):return {"skin_active":0,"bank_active":0,"pairs":[]}
 blo,bhi=bank.min((0,1)),bank.max((0,1));slo,shi=skin.min(1),skin.max(1)
 active=np.flatnonzero(np.all(shi>=blo,axis=1)&np.all(slo<=bhi,axis=1))
 if not len(active):return {"skin_active":0,"bank_active":0,"pairs":[]}
 rlo,rhi=slo[active].min(0),shi[active].max(0)
 bactive=np.flatnonzero(np.all(bank.max(1)>=rlo,axis=1)&np.all(bank.min(1)<=rhi,axis=1))
 sr=aud.records(skin[active],CHANGED[active],exact)
 br=aud.records(bank[bactive],bankids[bactive],exact)
 result=exact._audit_pair(sr,br,same_surface=False)
 return {"skin_active":int(len(active)),"bank_active":int(len(bactive)),"pairs":[[int(a),int(b)] for a,b in result.get("triangle_pairs",[])]}
def main():
 global CHANGED
 if OUT.exists() and any(OUT.iterdir()):raise FileExistsError("scoped exact output folder is not empty")
 pins={"base":sha(BASE),"candidate":sha(C),"candidate_report":sha(R),"layer":sha(L),"audit159":sha(AUDIT),"self_summary":sha(SELF),"self_pairs":sha(SELFP),"audit159_source":sha(AUD159),"audit130_source":sha(AUD130),"exact_predicate":sha(EXACT),"bank_helper":sha(BANKHELPER),"query_exporter":sha(EXPORT),"query_binary":sha(QUERYBIN)}
 if pins!=EXPECTED:raise ValueError(f"pinned input mismatch: {pins}")
 base=loadnpz(BASE);cand=loadnpz(C);ar=json.loads(AUDIT.read_text());sr=json.loads(SELF.read_text());stored_self=np.load(SELFP,allow_pickle=False);candidate_report=json.loads(R.read_text())
 base_hash=sha(BASE)
 if ar.get("candidate_sha256")!=base_hash or ar.get("complete") is not True or ar.get("expected_bank_count")!=185:raise ValueError("baseline audit159 receipt does not match complete pinned 185-bank candidate")
 if sr.get("candidate_sha256")!=base_hash or not sr.get("complete_surface_audit") or sr.get("exact_PLC_surface_pairs")!=0 or stored_self.shape!=(0,2):raise ValueError("baseline self receipt does not match pinned candidate")
 if candidate_report.get("candidate_base_sha256")!=base_hash or candidate_report.get("candidate_sha256")!=pins["candidate"]:raise ValueError("layer receipt does not pin the exact base/candidate")
 if candidate_report.get("selected_alpha")!=0.25 or not candidate_report.get("shape_gate_pass") or not candidate_report.get("no_new_affected_failure_identities_across_authored_orientation_nonpositive_and_conditioning_categories"):raise ValueError("candidate fails required shape/layer identity pre-gates")
 if set(base)!=set(cand):raise ValueError("candidate archive key set differs from base")
 for k in base:
  if k!="points_world_m" and (base[k].dtype!=cand[k].dtype or base[k].shape!=cand[k].shape or np.ascontiguousarray(base[k]).tobytes()!=np.ascontiguousarray(cand[k]).tobytes()):raise ValueError(f"unexpected non-point candidate mutation in {k}")
 fb=np.asarray(base["source_O_faces"],dtype=np.int64);fc=np.asarray(cand["source_O_faces"],dtype=np.int64)
 if not np.array_equal(fb,fc):raise ValueError("topology changed; this checker is point-only")
 bits0=np.asarray(base["points_world_m"],dtype=np.float32).view(np.uint32);bits1=np.asarray(cand["points_world_m"],dtype=np.float32).view(np.uint32)
 changed_nodes=np.flatnonzero(np.any(bits0!=bits1,axis=1)).astype(np.int64)
 if not len(changed_nodes):raise ValueError("candidate has no emitted Float32 movement")
 CHANGED=np.flatnonzero(np.any(np.isin(fb,changed_nodes),axis=1)).astype(np.int64)
 if candidate_report.get("changed_node_rows")!=changed_nodes.tolist() or candidate_report.get("changed_face_rows")!=CHANGED.tolist():raise ValueError("candidate receipt changed-node/face rows differ from emitted Float32 diff")
 if not candidate_report.get("typed_anchors_bitwise_preserved"):raise ValueError("typed anchor preservation failed in preflight receipt")
 candidate_nodes=np.asarray(cand["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
 if not np.array_equal(bits0[candidate_nodes],bits1[candidate_nodes]):raise ValueError("typed anchors changed in emitted Float32")
 if not CHANGED.size:raise ValueError("no changed face rows")
 if OUT.exists():raise FileExistsError("scoped exact output folder already exists")
 OUT.mkdir(parents=True)
 audit_mod=mod("audit159_local_subset",AUD159);exact=audit_mod.exact_module()
 helper=mod("bank_input_helper_local_subset",BANKHELPER)
 T,ids,gids,bodies,M,offset,pos,keys,input_pins,unused=helper.read_bank_inputs(BANKDIR,E)
 bankkeys=[(int(x[0]),int(x[1])) for x in keys]
 before_world=base["points_world_m"].astype(np.float64);after_world=cand["points_world_m"].astype(np.float64)
 before={};after={};before_pairs=set();after_pairs=set();per_bank=[]
 for body,bone in bankkeys:
  # Retain the full global skin-face identities with the audit owner predicate.
  bres=audit_changed_faces(before_world,fb,body,bone,T,ids,pos,M,offset,exact,audit_mod)
  ares=audit_changed_faces(after_world,fc,body,bone,T,ids,pos,M,offset,exact,audit_mod)
  bp={(body,bone,int(a),int(b)) for a,b in bres["pairs"]}
  ap={(body,bone,int(a),int(b)) for a,b in ares["pairs"]}
  before.update({(body,bone):bp});after.update({(body,bone):ap});before_pairs|=bp;after_pairs|=ap
  if bp or ap:per_bank.append({"body":body,"stable_bone_id":bone,"before_count":len(bp),"after_count":len(ap),"removed":sorted([list(x) for x in bp-ap]),"added":sorted([list(x) for x in ap-bp]),"before_pairs":[list(x) for x in sorted(bp)],"after_pairs":[list(x) for x in sorted(ap)],"aabb_active_before":{"skin":bres["skin_active"],"bank":bres["bank_active"]},"aabb_active_after":{"skin":ares["skin_active"],"bank":ares["bank_active"]}})
 # Check baseline local predicate identities against the full stored 159 receipt.
 expected_before=set()
 for bank in ar["banks"]:
  body,bone=int(bank["body"]),int(bank["stable_bone_id"])
  for a,b in ((bank.get("result") or {}).get("triangle_pairs") or []):
   if int(a) in set(CHANGED.tolist()):expected_before.add((body,bone,int(a),int(b)))
 if before_pairs!=expected_before:raise ValueError(f"local exact baseline does not reproduce audit159 witness identities: local={sorted(before_pairs)},stored={sorted(expected_before)}")
 new_pairs=after_pairs-before_pairs;removed_pairs=before_pairs-after_pairs
 # Canonical zero-motion control: the pinned baseline selected rows must be self-clean before candidate evaluation.
 base_X=np.asarray(base["points_world_m"],dtype=np.float32);base_records,_=exact.float32_surface_records(base_X,fb,source_seam_groups=())
 base_self_result=exact._audit_self_pairs_touching_faces(base_records,set(map(int,CHANGED)))
 base_self_pairs=[[int(a),int(b)] for a,b in (base_self_result.get("triangle_pairs") or [])]
 if base_self_pairs:raise ValueError(f"pinned baseline selected rows have exact self pairs: {base_self_pairs[:20]}")
 # Exact self audit on every changed face against the complete surface, preserving original vertex/face identities.
 X=np.asarray(cand["points_world_m"],dtype=np.float32);records,_=exact.float32_surface_records(X,fc,source_seam_groups=())
 self_result=exact._audit_self_pairs_touching_faces(records,set(map(int,CHANGED)))
 self_pairs=[[int(a),int(b)] for a,b in (self_result.get("triangle_pairs") or [])]
 # Query all four vertices used by the changed faces against all 185 exact bone solids.
 node_ids=np.unique(fc[CHANGED].reshape(-1)).astype(np.int64)
 if len(node_ids)==0:raise ValueError("no vertices in changed faces")
 qdir=OUT/"inside-query";qdir.mkdir(parents=True)
 qnpz=qdir/"points-world.npz";np.savez_compressed(qnpz,points_world_f32=X[node_ids])
 qpoints=qdir/"points.csv";qprepare=qdir/"prepare.log"
 prep_args=[sys.executable,str(EXPORT),"--candidate-npz",str(qnpz),"--input-sha256",sha(qnpz),"--field","points_world_f32","--output",str(qpoints)]
 cp=subprocess.run(prep_args,capture_output=True,text=True);qprepare.write_text(cp.stdout+cp.stderr)
 if cp.returncode:raise RuntimeError(f"query-point export failed: {cp.returncode}")
 qpairs=qdir/"pairs.csv";qnodes=qdir/"nodes.csv";qlog=qdir/"query.log"
 qargs=[str(QUERYBIN),str(BANKDIR/"exact-emitted.triangles.csv"),str(BANKDIR/"exact-emitted.bodies.csv"),str(qpoints),str(qpairs),str(qnodes),"185"]
 cp=subprocess.run(qargs,capture_output=True,text=True);qlog.write_text(cp.stdout+cp.stderr)
 if cp.returncode:raise RuntimeError(f"exact all-185 node query failed: {cp.returncode}")
 point_rows=list(csv.DictReader(qpoints.open()));node_rows=list(csv.DictReader(qnodes.open()));pair_rows=list(csv.DictReader(qpairs.open()))
 if not (len(point_rows)==len(node_rows)==len(node_ids)):raise ValueError("inside query output row count mismatch")
 bits=X.view(np.uint32)
 for i,(pr,nr) in enumerate(zip(point_rows,node_rows)):
  if int(pr["query_row"])!=i or int(nr["query_row"])!=i:raise ValueError("query row identity changed")
  if any(int(pr[k])!=int(bits[int(node_ids[i]),j]) for j,k in enumerate(("x_bits_u32","y_bits_u32","z_bits_u32"))):raise ValueError("query point differs from emitted Float32 node bits")
  if int(nr["total_bone_banks"])!=185 or int(nr["ambiguous_banks"])!=0 or int(nr["invalid_banks"])!=0:raise ValueError("affected node query incomplete or ambiguous")
  if sum(int(nr[k]) for k in ("aabb_proven_outside_banks","parity_outside_banks","inside_banks","ambiguous_banks","invalid_banks"))!=185:raise ValueError("affected node count not 185")
 current_inside=set()
 for r in pair_rows:
  if r["classification"] not in ("inside","outside") or int(r["ambiguous"])!=0:raise ValueError("invalid/ambiguous inside query pair")
  if r["classification"]=="inside":current_inside.add((int(node_ids[int(r["query_row"])]),int(r["bank_body"]),int(r["bank_stable_id"])))
 baseline_inside={(int(w["query_row"]),int(w["bank_body"]),int(w["bank_stable_id"])) for w in sr.get("inside_witnesses",[]) if int(w["query_row"]) in set(map(int,node_ids))}
 # Per-row ownership, shape, and identity checks.
 target_before={(a,b) for body,bone,a,b in before_pairs if body==65 and bone==99}
 target_after={(a,b) for body,bone,a,b in after_pairs if body==65 and bone==99}
 local_pass=bool(not new_pairs and len(after_pairs)<len(before_pairs) and len(self_pairs)==0 and current_inside==baseline_inside)
 status="LOCAL_DELTA_PASS_PENDING_FULL_AUDITS" if local_pass else "LOCAL_DELTA_FAIL_OR_NO_STRICT_IMPROVEMENT"
 report={
  "schema":"numi.human.b65-99.alpha025-exact-local.v1",
  "status":status,"local_delta_checks_pass":local_pass,"fit_shape_gate_pass":bool(candidate_report["shape_gate_pass"]),"layer_no_new_failure_identity_pre_gate":bool(candidate_report["no_new_affected_failure_identities_across_authored_orientation_nonpositive_and_conditioning_categories"]),
  "geometry_admitted":False,"native_admission":False,"accepted_physical_steps":0,
  "candidate_path":str(C),"candidate_sha256":EXPECTED["candidate"],"candidate_report_path":str(R),"candidate_report_sha256":EXPECTED["candidate_report"],"base_candidate_path":str(BASE),"base_candidate_sha256":EXPECTED["base"],
  "changed_source_face_rows_incident_to_changed_points":CHANGED.tolist(),"changed_point_rows":changed_nodes.tolist(),"face_topology_bitwise_unchanged":True,"all_non_point_archive_arrays_bitwise_unchanged":True,
  "exact_bone_scope":f"{len(CHANGED)} selected changed skin rows against every triangle in all 185 pinned typed bone banks; unchanged rows are inherited from complete baseline after bitwise-point and full-face-diff checks",
  "exact_bone_scope_changed_face_count":int(len(CHANGED)),
  "exact_bone_predicate_sha256":EXPECTED["exact_predicate"],"bank_input_identity_count":len(bankkeys),"expected_bank_count":185,
  "baseline_local_pairs_reproduced_from_audit159":True,"baseline_changed_face_pair_count":len(before_pairs),"post_changed_face_pair_count":len(after_pairs),
  "all_bank_new_pairs":sorted([list(x) for x in new_pairs]),"all_bank_removed_pairs":sorted([list(x) for x in removed_pairs]),
  "all_bank_no_new_pair":not new_pairs,"all_bank_strict_pair_reduction":len(after_pairs)<len(before_pairs),
  "target_65_99_before_pairs":sorted([list(x) for x in target_before]),"target_65_99_after_pairs":sorted([list(x) for x in target_after]),
  "target_65_99_before_count":len(target_before),"target_65_99_after_count":len(target_after),
  "per_bank_changed_face_results":per_bank,
  "exact_self_scope":"all selected changed rows against the complete 42,379-face output surface using exact Float32 lattice predicates and original global face/vertex identity",
  "self_baseline_control_selected_face_count":int(base_self_result["selected_face_count"]),"self_baseline_control_pairs":base_self_pairs,
  "self_selected_face_count":int(self_result["selected_face_count"]),"self_complete_surface_face_count":int(self_result["complete_surface_face_count"]),
  "self_pairs_touching_changed_faces":self_pairs,"self_pairs_touching_changed_face_count":len(self_pairs),
  "baseline_complete_self_pairs":int(len(stored_self)),"self_local_no_new_pair":len(self_pairs)==0,
  "inside_scope":f"all {len(node_ids)} unique vertices incident to changed rows, queried against all 185 typed bone solids with pinned static-query binary",
  "inside_query_node_ids":node_ids.tolist(),"inside_query_bank_count_per_node":[int(r["total_bone_banks"]) for r in node_rows],
  "inside_ambiguous_or_invalid_nodes":sum(int(int(r["ambiguous_banks"])!=0 or int(r["invalid_banks"])!=0) for r in node_rows),
  "inside_pairs_local_query":sorted([list(x) for x in current_inside]),"inside_pairs_stored_baseline":sorted([list(x) for x in baseline_inside]),
  "inside_matches_baseline":current_inside==baseline_inside,"changed_points_are_exact_f32_diff_rows":True,
  "source_detail_and_immediate_parent":"This is point-only: all topology and ancestry arrays are bitwise inherited from pinned a69.",
  "input_pins":pins,"layer_input_path":str(L),
  "audit_sources":{"audit159":str(AUD159),"audit130":str(AUD130),"exact_predicate":str(EXACT),"bank_helper":str(BANKHELPER),"query_exporter":str(EXPORT),"query_binary":str(QUERYBIN)},
  "limitations":["this is a two-face exact delta check, not a rerun of the full 42,379-face by 185-bank surface audit","inside classifications are local to the four vertices; all other vertex positions are bitwise unchanged from the complete pinned baseline receipt","edge clearance, solid separation, tissue thickness, physical transfer, and native admission remain unverified"]}
 (OUT/"bone-local.json").write_text(json.dumps({"baseline_pairs":sorted([list(x) for x in before_pairs]),"candidate_pairs":sorted([list(x) for x in after_pairs]),"new_pairs":sorted([list(x) for x in new_pairs]),"removed_pairs":sorted([list(x) for x in removed_pairs])},indent=2)+"\n")
 (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False,default=lambda v:v.item() if hasattr(v,"item") else v.tolist() if hasattr(v,"tolist") else str(v))+"\n")
 print(json.dumps({k:report[k] for k in ("status","baseline_changed_face_pair_count","post_changed_face_pair_count","all_bank_new_pairs","all_bank_removed_pairs","all_bank_no_new_pair","all_bank_strict_pair_reduction","target_65_99_before_count","target_65_99_after_count","self_pairs_touching_changed_face_count","inside_query_node_ids","inside_pairs_local_query","inside_pairs_stored_baseline","inside_matches_baseline")},indent=2,sort_keys=True))
if __name__=="__main__":main()
