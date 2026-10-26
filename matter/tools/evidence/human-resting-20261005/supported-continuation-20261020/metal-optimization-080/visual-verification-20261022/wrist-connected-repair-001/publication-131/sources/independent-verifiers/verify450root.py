"""Independent construction-only proof for fixed-guard backtrack 450.

Uses existing exact 130/159 reports and the saved 447 preprojection certificate;
no geometry audit, solver, or refreshed row construction is run here.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE=D/"D/seed-preserving-backtrack-436/run-001/backtrack-373/pass-015/candidate.npz"
RUN=D/"D/fixed-guard-backtrack-450/run-003/backtrack-373"
ROOT=RUN.parent
STATE=RUN/"selected-result.json"
FIT=D/"barycentric-connected-clearance-170/grouped-fit-440/run-001"
TRIAL=D/"D/grouped-backtrack-442/run-001/backtrack-373/pass-007/candidate.npz"
OUT=RUN/"root-independent-construction-verification.json"
SOURCE=D/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def owner401():
 p=D/"D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py";sys.path.insert(0,str(p.parent))
 s=importlib.util.spec_from_file_location("verify450_401",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def point_set(q):
 rows=q.get("inside_witnesses")
 if not isinstance(rows,list):raise AssertionError("point summary lacks inside witnesses")
 out={(int(r["query_row"]),int(r["bank_body"]),int(r["bank_stable_id"])) for r in rows}
 if len(out)!=len(rows):raise AssertionError("duplicate typed point identities")
 return out
def surface_set(r):
 if r.get("status")!="COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT" or r.get("complete") is not True:raise AssertionError("159 report is incomplete")
 banks=r.get("banks")
 if not isinstance(banks,list) or len(banks)!=185 or r.get("bank_count")!=185 or r.get("expected_bank_count")!=185:raise AssertionError("159 report does not bind all 185 banks")
 pairs=set();total=0
 for b in banks:
  result=b.get("result")
  if result is None:continue
  if not isinstance(result,dict):raise AssertionError("malformed 159 bank result")
  triangles=result.get("triangle_pairs")
  if not isinstance(triangles,list) or int(result.get("count",-1))!=len(triangles):raise AssertionError("159 pair count/list mismatch")
  total+=len(triangles)
  for face,tri in triangles:
   key=(int(face),int(b["body"]),int(b["stable_bone_id"]),int(tri))
   if key in pairs:raise AssertionError("duplicate typed surface identity")
   pairs.add(key)
 if len(pairs)!=total:raise AssertionError("159 typed identity count mismatch")
 return pairs,total

def main():
 state=read(STATE);candidate=Path(state["candidate"])
 if candidate.resolve()!=(RUN/"pass-004/candidate.npz").resolve() or sha(candidate)!="32daa86b90f18f19f20790c4bf64e5b772c9ee0b65646df48a724ffa5d70c5e0":raise AssertionError("selected 450 candidate pin mismatch")
 if state.get("candidate_sha256")!=sha(candidate):raise AssertionError("450 state candidate hash mismatch")
 source_inputs=read(RUN/"source-inputs.json")
 if source_inputs.get("start")!=str(BASE) or source_inputs.get("start_sha256")!=sha(BASE):raise AssertionError("450 start is not pinned 436 candidate")
 if source_inputs.get("trial")!=str(TRIAL) or source_inputs.get("trial_sha256")!=sha(TRIAL):raise AssertionError("450 trial is not selected 442 candidate")
 if state.get("parent_fit")!=str(FIT/"report.json"):raise AssertionError("450 parent fit is not grouped-fit-440")
 a=np.load(BASE,allow_pickle=False);b=np.load(candidate,allow_pickle=False)
 if set(a.files)!=set(b.files):raise AssertionError("candidate array key set changed")
 for k in a.files:
  if k=="points_world_m":continue
  if a[k].dtype!=b[k].dtype or a[k].shape!=b[k].shape or a[k].tobytes()!=b[k].tobytes():raise AssertionError("non-coordinate metadata changed: "+k)
 anchors=np.asarray(a["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
 if len(anchors)!=237 or not np.array_equal(a["points_world_m"][anchors],b["points_world_m"][anchors]):raise AssertionError("237 anchors changed")
 raw=np.load(FIT/"raw-lp-candidate.npz",allow_pickle=False)
 support=np.asarray(raw["local_witness_cluster_222_lp_support_rows"],dtype=np.int64)
 if support.ndim!=1 or len(np.unique(support))!=len(support) or np.any(support<0) or np.any(support>=len(a["points_world_m"])):raise AssertionError("440 fit support is malformed")
 outside=np.setdiff1d(np.arange(len(a["points_world_m"]),dtype=np.int64),support)
 if not np.array_equal(a["points_world_m"][outside],b["points_world_m"][outside]):raise AssertionError("450 changed points outside 440 raw support")

 m=owner401();source_ref,source_receipt=m.compose_source_reference(a,SOURCE)
 x=np.asarray(a["points_world_m"],dtype=np.float32);y=np.asarray(b["points_world_m"],dtype=np.float32)
 step=np.linalg.norm(y.astype(np.float64)-x.astype(np.float64),axis=1)*1000.0
 source_disp=np.linalg.norm(y.astype(np.float64)-source_ref,axis=1)*1000.0
 if step.max(initial=0.0)>1+1e-6 or source_disp.max(initial=0.0)>12+1e-6:raise AssertionError("450 step/source construction cap failed")
 area=m.full_f32_area_gate(y,x.astype(np.float64)*1000,
  np.asarray(a["area_reference_037_points_world_m_f64"],dtype=np.float64)*1000,
  a["source_O_faces"],.1)
 if not area.get("pass"):raise AssertionError("450 exact Float32 area gate failed")

 point_path=Path(state["point_summary"]);surface_path=Path(state["surface_report"])
 q=read(point_path);s=read(surface_path);candidate_sha=sha(candidate)
 if state.get("point_summary_sha256")!=sha(point_path):raise AssertionError("450 point summary hash mismatch")
 if q.get("candidate_sha256")!=candidate_sha or s.get("candidate_sha256")!=candidate_sha:raise AssertionError("450 exact reports candidate mismatch")
 if q.get("exact_PLC_surface_pairs")!=0 or q.get("complete_surface_audit") is not True:raise AssertionError("450 point report self-audit incomplete/nonzero")
 final_points=point_set(q);final_surface,ns=surface_set(s)
 base_run=D/"D/seed-preserving-backtrack-436/run-001/backtrack-373/pass-015"
 qb=read(base_run/"audit-130/summary.json");sb=read(base_run/"audit-159/report.json")
 if qb.get("candidate_sha256")!=sha(BASE) or sb.get("candidate_sha256")!=sha(BASE):raise AssertionError("436 exact reports do not bind baseline")
 base_points=point_set(qb);base_surface,nbase=surface_set(sb)
 point_intro=final_points-base_points;point_resolved=base_points-final_points
 surface_intro=final_surface-base_surface;surface_resolved=base_surface-final_surface
 result=read(RUN/"pass-004/result.json");pm=result["point_merit"];jm=result["joint_merit"];dm=result["depth_merit"]
 if len(base_points)!=322 or len(final_points)!=322 or len(base_surface)!=3225 or ns!=3153:raise AssertionError("450 exact reported counts differ from 436 comparison")
 if (pm["before_inside_pairs"]!=len(base_points) or pm["after_inside_pairs"]!=len(final_points)
  or jm["before_surface_pairs"]!=nbase or jm["after_surface_pairs"]!=ns
  or dm["introduced_surface_pair_count"]!=len(surface_intro)):
  raise AssertionError("450 merit fields disagree with exact 130/159 pair sets")

 pre=read(ROOT/"preprojection.json");scale=read(ROOT/"seed-scale-report.json")
 cert=pre.get("seed_cut_lp",{}).get("fixed_guard_certificate",{})
 if pre.get("status")!="GRADED_426_WITH_FULL_SELF_SEEDS_AND_FIXED_BONE_GUARDS_447" or scale.get("status")!=pre.get("status"):
  raise AssertionError("450 did not use pinned 447 wrapper")
 if not cert.get("passed") or cert.get("row_count")!=12678 or cert.get("violated_rows")!=0 or cert.get("anchor_max_abs_scale")!=0:
  raise AssertionError("450 original 313 row/anchor certificate failed")
 receipt=cert.get("input_receipt",{})
 npz=Path(receipt.get("npz_path",""));report=Path(receipt.get("report_path",""))
 if not npz.is_file() or sha(npz)!=receipt.get("npz_sha256") or not report.is_file() or sha(report)!=receipt.get("report_sha256"):
  raise AssertionError("450 449 guard source receipt byte pins fail")

 out={"status":"ROOT_INDEPENDENT_CONSTRUCTION_PASS","candidate":str(candidate),"candidate_sha256":candidate_sha,
  "selected_state_sha256":sha(STATE),"base_candidate":str(BASE),"base_candidate_sha256":sha(BASE),
  "fit_trial_candidate":str(TRIAL),"fit_trial_candidate_sha256":sha(TRIAL),
  "fit_440_candidate_sha256":sha(FIT/"candidate.npz"),"fit_440_raw_candidate_sha256":sha(FIT/"raw-lp-candidate.npz"),
  "fit_440_report_sha256":sha(FIT/"report.json"),"fit_440_raw_support_rows":int(len(support)),
  "fit_440_raw_support_sha256":hashlib.sha256(np.ascontiguousarray(support,dtype="<i8").tobytes()).hexdigest(),
  "outside_fit_support_unchanged":True,"source_arrays_bitwise_unchanged":len(a.files)-1,
  "anchors_237_bitwise_unchanged":True,"changed_nodes":int(np.count_nonzero(step)),
  "max_step_mm":float(step.max(initial=0)),"max_source_displacement_mm":float(source_disp.max(initial=0)),
  "area_gate":area,"source_reference_receipt":source_receipt,
  "point_pairs_before":len(base_points),"point_pairs_after":len(final_points),
  "point_pairs_resolved":len(point_resolved),"point_pairs_introduced":len(point_intro),
  "max_point_depth_before_m":pm["before_maximum_depth_m"],"max_point_depth_after_m":pm["after_maximum_depth_m"],
  "squared_depth_before_m2":pm["before_squared_depth_m2"],"squared_depth_after_m2":pm["after_squared_depth_m2"],
  "surface_pairs_before":nbase,"surface_pairs_after":ns,
  "surface_pairs_resolved":len(surface_resolved),"surface_pairs_introduced":len(surface_intro),
  "total_pairs_before":jm["before_total_pairs"],"total_pairs_after":jm["after_total_pairs"],
  "complete_185_bank_surface_receipt":True,"typed_bank_count":185,"self_pairs":0,
  "preprojection_sha256":sha(ROOT/"preprojection.json"),"launch_sha256":sha(ROOT/"launch.json"),
  "seed_scale_report_sha256":sha(ROOT/"seed-scale-report.json"),
  "fixed_guard_certificate":{k:cert.get(k) for k in ("passed","row_count","checked_rows","violated_rows","max_violation_mm","tolerance_mm","anchor_max_abs_scale")},
  "fixed_guard_receipt":{k:receipt.get(k) for k in ("npz_path","npz_sha256","report_path","report_sha256","base_candidate_sha256","typed_pair_count","linear_row_count","313_owner_sha256")},
  "point_summary_sha256":sha(point_path),"surface_report_sha256":sha(surface_path),
  "base_point_summary_sha256":sha(base_run/"audit-130/summary.json"),"base_surface_report_sha256":sha(base_run/"audit-159/report.json"),
  "native_admission":False,"accepted_physical_steps":0,"geometry_admission":False,"zero_clearance_achieved":False}
 OUT.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n",encoding="utf-8");print(json.dumps(out,sort_keys=True,indent=2))
if __name__=="__main__":main()
