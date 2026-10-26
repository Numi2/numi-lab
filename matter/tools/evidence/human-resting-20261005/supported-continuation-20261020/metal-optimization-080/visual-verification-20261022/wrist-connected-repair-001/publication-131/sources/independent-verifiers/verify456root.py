"""Independent construction-only proof for constraint-project backtrack 456.

Uses the existing selected candidate, exact 130/159 receipts, saved 447
preprojection certificate, and 440+445 support arrays. It runs no solver or
geometry audit and keeps the 455 and 456 trials distinct.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import numpy as np
import importlib.util,sys
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE=D/"D/fixed-guard-backtrack-450/run-003/backtrack-373/pass-004/candidate.npz"
TRIAL=D/"D/point-first-backtrack-446/run-001/backtrack-373/pass-015/candidate.npz"
RUN=D/"D/constraint-project-backtrack-456/run-001/backtrack-373"
ROOT=RUN.parent;STATE=RUN/"selected-result.json";OUT=RUN/"root-independent-construction-verification.json"
FIT440=D/"barycentric-connected-clearance-170/grouped-fit-440/run-001"
FIT445=D/"barycentric-connected-clearance-170/point-first-fit-445/run-001"
SOURCE=D/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def owner401():
 p=D/"D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py";sys.path.insert(0,str(p.parent))
 s=importlib.util.spec_from_file_location("verify455_401",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def points_set(q):
 rows=q.get("inside_witnesses")
 if not isinstance(rows,list):raise AssertionError("130 report lacks inside witness set")
 out={(int(r["query_row"]),int(r["bank_body"]),int(r["bank_stable_id"])) for r in rows}
 if len(out)!=len(rows):raise AssertionError("duplicate typed 130 identities")
 return out
def surfaces_set(r):
 if r.get("status")!="COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT" or r.get("complete") is not True:raise AssertionError("159 report incomplete")
 banks=r.get("banks")
 if not isinstance(banks,list) or len(banks)!=185 or r.get("bank_count")!=185 or r.get("expected_bank_count")!=185:raise AssertionError("159 report missing complete 185-bank coverage")
 out=set();count=0
 for bank in banks:
  result=bank.get("result")
  if result is None:continue
  pairs=result.get("triangle_pairs")
  if not isinstance(pairs,list) or int(result.get("count",-1))!=len(pairs):raise AssertionError("159 bank result count mismatch")
  count+=len(pairs)
  for face,tri in pairs:
   pair=(int(face),int(bank["body"]),int(bank["stable_bone_id"]),int(tri))
   if pair in out:raise AssertionError("duplicate typed 159 identity")
   out.add(pair)
 if len(out)!=count:raise AssertionError("159 typed pair total mismatch")
 return out,count
def main():
 state=read(STATE);candidate=Path(state["candidate"])
 if candidate.resolve()!=(RUN/"pass-006/candidate.npz").resolve() or sha(candidate)!="1ac004ef6f6646ce4720360c50b7fd9855336a4967c3e97b199cf9bc0350049a":raise AssertionError("455 selected candidate path/hash mismatch")
 inputs=read(RUN/"source-inputs.json")
 if inputs.get("start")!=str(BASE) or inputs.get("start_sha256")!=sha(BASE):raise AssertionError("455 source base is not exact 450 pass-004")
 if inputs.get("trial")!=str(TRIAL) or inputs.get("trial_sha256")!=sha(TRIAL):raise AssertionError("455 trial is not exact selected 446 pass-015")
 a=np.load(BASE,allow_pickle=False);b=np.load(candidate,allow_pickle=False)
 if set(a.files)!=set(b.files):raise AssertionError("455 candidate array key set changed")
 for k in a.files:
  if k=="points_world_m":continue
  if a[k].dtype!=b[k].dtype or a[k].shape!=b[k].shape or a[k].tobytes()!=b[k].tobytes():raise AssertionError("455 non-coordinate source array changed: "+k)
 anchors=np.asarray(a["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
 if len(anchors)!=237 or not np.array_equal(a["points_world_m"][anchors],b["points_world_m"][anchors]):raise AssertionError("455 237 typed anchors changed")

 z440=np.load(FIT440/"raw-lp-candidate.npz",allow_pickle=False);z445=np.load(FIT445/"raw-lp-candidate.npz",allow_pickle=False)
 s440=np.asarray(z440["local_witness_cluster_222_lp_support_rows"],dtype=np.int64)
 s445=np.asarray(z445["local_witness_cluster_222_lp_support_rows"],dtype=np.int64)
 if len(np.unique(s440))!=len(s440) or len(np.unique(s445))!=len(s445):raise AssertionError("fit support arrays contain duplicate rows")
 support=np.union1d(s440,s445)
 cfg=read(ROOT/"backtrack-config.json")
 if not np.array_equal(np.asarray(cfg.get("fit_support_rows"),dtype=np.int64),support):raise AssertionError("455 config support is not exact 440+445 union")
 if np.any(support<0) or np.any(support>=len(a["points_world_m"])):raise AssertionError("455 union support outside node domain")
 outside=np.setdiff1d(np.arange(len(a["points_world_m"]),dtype=np.int64),support)
 if not np.array_equal(a["points_world_m"][outside],b["points_world_m"][outside]):raise AssertionError("455 changed nodes outside union support")

 m=owner401();source_ref,source_receipt=m.compose_source_reference(a,SOURCE)
 x=np.asarray(a["points_world_m"],dtype=np.float32);y=np.asarray(b["points_world_m"],dtype=np.float32)
 step=np.linalg.norm(y.astype(np.float64)-x.astype(np.float64),axis=1)*1000
 source_disp=np.linalg.norm(y.astype(np.float64)-source_ref,axis=1)*1000
 if step.max(initial=0)>1+1e-6 or source_disp.max(initial=0)>12+1e-6:raise AssertionError("455 step/source cap failed")
 area=m.full_f32_area_gate(y,x.astype(np.float64)*1000,
  np.asarray(a["area_reference_037_points_world_m_f64"],dtype=np.float64)*1000,
  a["source_O_faces"],.1)
 if not area.get("pass"):raise AssertionError("455 emitted Float32 area gate failed")

 point_path=Path(state["point_summary"]);surface_path=Path(state["surface_report"]);q=read(point_path);sr=read(surface_path)
 if state.get("point_summary_sha256")!=sha(point_path):raise AssertionError("455 point summary hash mismatch")
 if q.get("candidate_sha256")!=sha(candidate) or sr.get("candidate_sha256")!=sha(candidate):raise AssertionError("455 exact receipts candidate mismatch")
 if q.get("exact_PLC_surface_pairs")!=0 or q.get("complete_surface_audit") is not True:raise AssertionError("455 self audit incomplete/nonzero")
 fp=points_set(q);fs,nf=surfaces_set(sr)
 base_run=D/"D/fixed-guard-backtrack-450/run-003/backtrack-373/pass-004"
 qb=read(base_run/"audit-130/summary.json");sb=read(base_run/"audit-159/report.json")
 if qb.get("candidate_sha256")!=sha(BASE) or sb.get("candidate_sha256")!=sha(BASE):raise AssertionError("450 exact base reports do not bind")
 bp=points_set(qb);bs,nb=surfaces_set(sb)
 pintro=fp-bp;presolved=bp-fp;sintro=fs-bs;sresolved=bs-fs
 if len(bp)!=322 or len(fp)!=321 or nb!=3153 or nf!=3136:raise AssertionError("455 exact pair counts differ from 450 comparison")
 result=read(RUN/"pass-006/result.json");pm=result["point_merit"];jm=result["joint_merit"];dm=result["depth_merit"]
 if (pm["before_inside_pairs"]!=len(bp) or pm["after_inside_pairs"]!=len(fp)
  or jm["before_surface_pairs"]!=nb or jm["after_surface_pairs"]!=nf
  or dm["introduced_surface_pair_count"]!=len(sintro)
  or {tuple(x) for x in pm["introduced_typed_pairs"]}!=pintro):
  raise AssertionError("455 selected merit receipt disagrees with exact pair sets")
 if len(sintro)!=6:raise AssertionError("455 introduced surface identity count is not six")

 pre=read(ROOT/"preprojection.json");scale=read(ROOT/"seed-scale-report.json");pcert=pre.get("seed_cut_lp",{}).get("fixed_guard_certificate",{})
 if pre.get("status")!="CONSTRAINT_PROJECTION_WITHOUT_WITNESS_HALVING" or scale.get("status")!="GRADED_426_WITH_FULL_SELF_SEEDS_AND_FIXED_BONE_GUARDS_447":
  raise AssertionError("456 did not use direct 447 fixed guard projection")
 if not pcert.get("passed") or pcert.get("row_count")!=13494 or pcert.get("violated_rows")!=0 or pcert.get("anchor_max_abs_scale")!=0:
  raise AssertionError("455 313 guard row/anchor certificate failed")
 receipt=pcert.get("input_receipt",{});npz=Path(receipt.get("npz_path",""));report=Path(receipt.get("report_path",""))
 if not npz.is_file() or sha(npz)!=receipt.get("npz_sha256") or not report.is_file() or sha(report)!=receipt.get("report_sha256"):
  raise AssertionError("455 refresh454 input pins do not match their bytes")

 out={"status":"ROOT_INDEPENDENT_CONSTRUCTION_PASS","candidate":str(candidate),"candidate_sha256":sha(candidate),
  "selected_state_sha256":sha(STATE),"base_candidate":str(BASE),"base_candidate_sha256":sha(BASE),
  "trial_candidate":str(TRIAL),"trial_candidate_sha256":sha(TRIAL),
  "fit440_raw_candidate_sha256":sha(FIT440/"raw-lp-candidate.npz"),"fit445_raw_candidate_sha256":sha(FIT445/"raw-lp-candidate.npz"),
  "fit_support_440_rows":int(len(s440)),"fit_support_440_sha256":hashlib.sha256(np.ascontiguousarray(s440,dtype="<i8").tobytes()).hexdigest(),
  "fit_support_445_rows":int(len(s445)),"fit_support_445_sha256":hashlib.sha256(np.ascontiguousarray(s445,dtype="<i8").tobytes()).hexdigest(),
  "fit_support_union_rows":int(len(support)),"fit_support_union_sha256":hashlib.sha256(np.ascontiguousarray(support,dtype="<i8").tobytes()).hexdigest(),
  "outside_support_unchanged":True,"source_arrays_bitwise_unchanged":len(a.files)-1,"anchors_237_bitwise_unchanged":True,
  "changed_nodes":int(np.count_nonzero(step)),"max_step_mm":float(step.max(initial=0)),"max_source_displacement_mm":float(source_disp.max(initial=0)),
  "area_gate":area,"source_reference_receipt":source_receipt,
  "point_pairs_before":len(bp),"point_pairs_after":len(fp),"point_pairs_resolved":len(presolved),"point_pairs_introduced":len(pintro),
  "max_point_depth_before_m":pm["before_maximum_depth_m"],"max_point_depth_after_m":pm["after_maximum_depth_m"],
  "squared_depth_before_m2":pm["before_squared_depth_m2"],"squared_depth_after_m2":pm["after_squared_depth_m2"],
  "surface_pairs_before":nb,"surface_pairs_after":nf,"surface_pairs_resolved":len(sresolved),"surface_pairs_introduced":len(sintro),
  "total_pairs_before":jm["before_total_pairs"],"total_pairs_after":jm["after_total_pairs"],"complete_185_bank_surface_receipt":True,
  "typed_bank_count":185,"self_pairs":0,"fixed_guard_certificate":{k:pcert.get(k) for k in ("passed","row_count","checked_rows","violated_rows","max_violation_mm","tolerance_mm","anchor_max_abs_scale")},
  "fixed_guard_receipt":{k:receipt.get(k) for k in ("npz_path","npz_sha256","report_path","report_sha256","base_candidate_sha256","typed_pair_count","linear_row_count","313_owner_sha256")},
  "preprojection_sha256":sha(ROOT/"preprojection.json"),"launch_sha256":sha(ROOT/"launch.json"),
  "seed_scale_report_sha256":sha(ROOT/"seed-scale-report.json"),"point_summary_sha256":sha(point_path),"surface_report_sha256":sha(surface_path),
  "base_point_summary_sha256":sha(base_run/"audit-130/summary.json"),"base_surface_report_sha256":sha(base_run/"audit-159/report.json"),
  "native_admission":False,"accepted_physical_steps":0,"geometry_admission":False,"zero_clearance_achieved":False,
  "cross_candidate_merge_claim":False,"trial_455_candidate":"/Users/numi/Work/numi-human-connected-surface-069/D/merged-guard-backtrack-455/run-002/backtrack-373/pass-008/candidate.npz","trial_455_candidate_sha256":"1c5a6544948adf9840a274208c7a6be89c457947a3a4ac4d54fa802a63b1de59","preprojection_mode":"direct 447 projection without initial 423 witness halving"}
 OUT.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n",encoding="utf-8");print(json.dumps(out,sort_keys=True,indent=2))
if __name__=="__main__":main()
