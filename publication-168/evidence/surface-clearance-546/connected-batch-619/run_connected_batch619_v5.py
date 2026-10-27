#!/usr/bin/env python3
"""Bounded repeated 613/603 continuation; construction evidence only, never launched by import.

The opt-in partial route accepts only an existing, receipt-validated 564 zero-fallback
candidate after LP_NO_PROPOSAL. It never claims global LP convergence.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, shutil, sys
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
D=ROOT/"D/surface-clearance-546"
PY=Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
DEPTH=ROOT/"D/point-depth-refresh-413/refresh_point_depth_rows.py"
DEPTH_SHA="ee327a03af99bcb9b89f6344d1f3a7d6ead074cbf489fd985f04c22362ecff3f"
DEPTH_OWNER=ROOT/"D/point-depth-barriers-357/point_depth_barriers_357.py"
DEPTH_OWNER_SHA="dc8078515dd12b2edc62c0bc75759a7cb9472c9914e4704ecd41e04c4d5378e1"
PREP613=D/"connected-continuation-613/prepare_matched613.py"
STAGE546=D/"continue_surface_round546.py"
FIT603=D/"component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"
FIT603_HELPER=FIT603.with_name("component_scp_backtracking603.py")
FROZEN_V4=Path(__file__).with_name("run_connected_batch619_v4.py")
FROZEN_V4_SHA="bcc61255e76fe844754d4f2f066da248bb1a46457da728b75066a1ba4edef9f8"
PARTIAL564=D/"component-scp-area-backtrack-603/frozen-v1/component_lp_partial_timeout564.py"
RAW563=D/"raw-joint-capture-563/raw_joint_capture563.py"
CAPTURE546=D/"raw-lp-direction-capture-546/capture_raw_lp_directions546.py"
MATCHED583=D/"matched-connected-fit-583/run_matched_fit583.py"
FIT490=ROOT/"D/D/component-lp-decomposition-490/run_matched_component_fit_490.py"
FIT_TEMPLATE=D/"connected-continuation-584/run-001/round-001/emitted-fit/fit-argv.json"
MATCHED_TEMPLATE=D/"connected-continuation-584/run-001/round-001/emitted-fit/matched-inputs.json"
CHECK545=ROOT/"D/contained-clearance-545/run_candidate_checks545.py"
MERGE609=D/"self-seed-merge-609/merge_self_seeds609_v2.py"
RETENTION582=D/"local-witness-retention-582/retain_local_witnesses582.py"
FINAL546=D/"finalize_existing_checks546.py"
MIN_FIT_FREE=250*1024*1024
MIN_RETENTION_FREE=165*1024*1024
MAX_EXTRA_CUT_FITS=2
TRUST_MM="4"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):
    x=json.loads(Path(p).read_text())
    if not isinstance(x,dict): raise ValueError("expected JSON object: "+str(p))
    return x
def writej(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def loadmod(name,p):
    s=importlib.util.spec_from_file_location(name,p)
    if s is None or s.loader is None: raise ImportError(str(p))
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
M=loadmod("continue_round546_619",STAGE546)

def emit(event,**fields):
    print(json.dumps({"event":event,**fields},sort_keys=True,separators=(",",":")),flush=True)
def paths_exist(paths):
    missing=[str(p) for p in paths if not Path(p).is_file()]
    if missing: raise FileNotFoundError("missing inputs: "+", ".join(missing))
def disk_guard(free,minimum):
    if type(free) is not int or free<minimum: raise OSError(f"free disk {free} < required {minimum}")
def additional_cut_action(free):
    if type(free) is not int: raise ValueError("free disk must be integer bytes")
    if free>=MIN_FIT_FREE: return "cut"
    if free>=MIN_RETENTION_FREE: return "retain-last-fit"
    return "stop"
def replace_value(argv,flag,value):
    if argv.count(flag)!=1: raise ValueError("expected exactly one "+flag)
    i=argv.index(flag)
    if i+1>=len(argv) or argv[i+1].startswith("--"): raise ValueError("missing value for "+flag)
    argv[i+1]=str(value)
def set_scp1(argv,output):
    a=list(argv)
    for f in ("--area-scp-iterations","--time-limit-s","--base-candidate","--self-pairs","--output"):
        if a.count(f)!=1: raise ValueError("fit argv needs exactly one "+f)
    if Path(a[0])!=PY or Path(a[1])!=FIT603: raise ValueError("fit is not frozen 603")
    if float(a[a.index("--time-limit-s")+1])!=90.: raise ValueError("fit budget must remain 90 seconds")
    replace_value(a,"--area-scp-iterations","1");replace_value(a,"--output",output)
    return a
def validate_baseline_reports(point_report,surface_report,candidate_hash):
    if point_report.get("candidate_sha256")!=candidate_hash or point_report.get("complete_surface_audit") is not True: raise ValueError("incomplete/stale baseline 130")
    if type(point_report.get("exact_PLC_surface_pairs")) is not int or point_report["exact_PLC_surface_pairs"]!=0: raise ValueError("baseline must have zero exact self pairs")
    if type(point_report.get("inside_node_bank_pairs")) is not int or point_report["inside_node_bank_pairs"]!=0: raise ValueError("positive-inside baselines are unsupported")
    if surface_report.get("candidate_sha256")!=candidate_hash or surface_report.get("complete") is not True or surface_report.get("expected_bank_count")!=185 or surface_report.get("bank_count")!=185 or len(surface_report.get("banks",[]))!=185: raise ValueError("baseline needs complete 185-bank surface report")
def strict_baseline(state_path):
    s=readj(state_path)
    if s.get("native_admission") is not False or s.get("accepted_physical_steps")!=0: raise ValueError("state is not construction-only")
    roles={k:Path(s[k]) for k in ("candidate","point_summary","surface_report")}
    for k,p in roles.items():
        if sha(p)!=s.get(k+"_sha256"): raise ValueError("state hash mismatch: "+k)
    p=readj(roles["point_summary"]);b=readj(roles["surface_report"]);h=s["candidate_sha256"]
    validate_baseline_reports(p,b,h)
    pair=roles["point_summary"].parent/"surface-self-pairs.npy"
    arr=np.load(pair,allow_pickle=False)
    if arr.shape!=(0,2) or arr.dtype.kind not in "iu": raise ValueError("baseline pair file must be integer (0,2)")
    query=roles["point_summary"].parent/"query/query-receipt.json"
    paths_exist((pair,query))
    return s,roles,pair,query

def is_exact_self_failure(report,audit_path,pairs_path,candidate_hash):
    if report.get("status")!="FAILED" or report.get("error")!="exact self intersections": return False
    if not Path(audit_path).is_file() or not Path(pairs_path).is_file(): return False
    audit=readj(audit_path);pairs=np.load(pairs_path,allow_pickle=False);n=audit.get("exact_PLC_surface_pairs")
    return (audit.get("candidate_sha256")==candidate_hash and audit.get("complete_surface_audit") is True
            and type(n) is int and n>0 and pairs.dtype.kind in "iu" and pairs.shape==(n,2))

def owners():
    q=(Path(__file__),DEPTH,DEPTH_OWNER,PREP613,STAGE546,
       D/"refresh_surface_constraints546.py",D/"refresh_guards_multi_parent_546.py",
       D/"source-current-targets-597/build_joint_outward_targets597_v4.py",
       D/"source-current-targets-597/angular_joint_direction597.py",
       D/"refined-fit-604/verify_joint_targets604.py",
       D/"connected-fit-598/target-compatibility-001/compatibility_report598_v2.py",
       D/"joint-refined-595/joint_refined595.py",
       FIT490,FIT603,FIT603_HELPER,CHECK545,MERGE609,RETENTION582,
       FROZEN_V4,PARTIAL564,RAW563,CAPTURE546,MATCHED583,
       D/"local-witness-retention-567/retain_local_witnesses567.py",
       ROOT/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py",
       FINAL546,ROOT/"D/joint-clearance-merit-175/canonical/matter/tools/human_skin_fat_bone_audit.py",
       ROOT/"publication-101/evidence/204/source/fit/self_separation_026.py",
       ROOT/"D/focused-clearance-541/constraints/refresh_guards_541.py",
       ROOT/"publication-113/sources/connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py",
       ROOT/"publication-113/sources/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py",
       ROOT/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz",
       FIT_TEMPLATE,MATCHED_TEMPLATE)
    paths_exist(q)
    if sha(DEPTH)!=DEPTH_SHA or sha(DEPTH_OWNER)!=DEPTH_OWNER_SHA: raise ValueError("413 owner pin mismatch")
    return q
def verify_pins(pins):
    changed=[p for p,h in pins.items() if not Path(p).is_file() or sha(p)!=h]
    if changed: raise RuntimeError("source/input changed: "+", ".join(changed))
def stage(argv,stem,*,slot,sources,timeout):
    return M.run_stage(argv,stem,slot=slot,source_paths=tuple(sources),timeout=timeout)

def recoverable_self_check(execution,report,audit,pairs,candidate_hash):
    return (execution.get("returncode")!=0 and execution.get("source_pins_unchanged") is True
            and is_exact_self_failure(report,audit,pairs,candidate_hash))
def build_check545_argv(candidate,baseline,out):
    return [PY,CHECK545,"--candidate",candidate,"--baseline-state",baseline,
            "--trust-mm",TRUST_MM,"--output-dir",out]
def build_retention582_argv(state,target,out):
    return [PY,RETENTION582,"--state",state,"--target",target,"--out",out,
            "--trust-mm",TRUST_MM,"--max-edge-gradient",".5"]
def build_finalizer546_argv(candidate,baseline,point_summary,surface_report,out):
    return [PY,FINAL546,"--candidate",candidate,"--baseline-state",baseline,
            "--point-summary",point_summary,"--surface-report",surface_report,
            "--trust-mm",TRUST_MM,"--out",out]
def _need(condition,message):
    if not condition: raise ValueError(message)

def _same_array(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def validate_certified_partial_candidate(fit_dir,base_path,fit_report):
    """Validate an existing 564 zero-fallback direction; never infer LP convergence."""
    fit_dir=Path(fit_dir).resolve();base_path=Path(base_path).resolve()
    base_hash=sha(base_path)
    _need(fit_report.get("status")=="LP_NO_PROPOSAL","partial route requires preserved LP_NO_PROPOSAL report")
    _need(fit_report.get("solver",{}).get("success") is not True,"partial report claims solver success")
    fit_inputs=fit_report.get("inputs",{})
    _need(fit_inputs.get("base_candidate_sha256")==base_hash,"partial fit report base hash mismatch")
    _need(Path(fit_inputs.get("base_candidate","")).resolve()==base_path,"partial fit report base path mismatch")
    partial=fit_report.get("full_coordinate_partial_timeout564")
    _need(isinstance(partial,dict),"missing 564 partial-timeout receipt")
    for key in ("construction_only","partial_timeout_is_solver_failure_not_global_success",
                "partial_timeout_zero_fallback_opt_in","raw_joint_directions_are_unadmitted"):
        _need(partial.get(key) is True,"564 receipt missing required true flag: "+key)
    _need(partial.get("native_admission") is False and partial.get("accepted_physical_steps")==0,
          "partial receipt is not construction-only")
    _need(partial.get("partial_timeout_direction_count")==1 and
          partial.get("raw_joint_capture_captured_delta_count")==1 and
          partial.get("raw_joint_capture_record_count")==1,
          "expected exactly one captured partial direction")
    _need(partial.get("raw_joint_capture_status")=="RAW_JOINT_LP_DIRECTIONS_CAPTURED_UNADMITTED",
          "partial raw capture status mismatch")
    captured_base=partial.get("captured_basis_summary",{})
    _need(Path(captured_base.get("base_candidate","")).resolve()==base_path and
          captured_base.get("base_sha256")==base_hash,
          "captured basis does not match requested base")
    pins=partial.get("pinned_owner_sha256",{})
    source_expectations={
        "563_raw_capture_helper":RAW563,
        "564_partial_timeout_component_adapter":PARTIAL564,
        "603_component_scp_backtracking_helper":FIT603_HELPER,
        "matched_490_entry":FIT490,
    }
    for role,path in source_expectations.items():
        _need(Path(path).is_file() and pins.get(role)==sha(path),"564 source pin mismatch: "+role)
    _need(partial.get("active_component_adapter_source_sha256")==sha(PARTIAL564),
          "active 564 adapter source hash mismatch")
    _need(partial.get("capture_helper_source_sha256")==sha(RAW563),
          "active 563 capture source hash mismatch")
    _need(partial.get("source_sha256")==sha(FIT603),"fit source hash mismatch")

    summary_path=Path(partial.get("raw_joint_capture_summary_path","")).resolve()
    expected_summary=(fit_dir/"raw-joint-capture-563/report.json").resolve()
    _need(summary_path==expected_summary and summary_path.is_file(),"raw capture summary path mismatch/missing")
    _need(sha(summary_path)==partial.get("raw_joint_capture_summary_sha256"),"raw capture summary hash mismatch")
    capture=readj(summary_path)
    _need(capture.get("schema")=="numi.human.raw_joint_lp_direction_capture563.v1" and
          capture.get("status")=="RAW_JOINT_LP_DIRECTIONS_CAPTURED_UNADMITTED",
          "raw capture receipt status/schema mismatch")
    _need(capture.get("native_admission") is False and capture.get("accepted_physical_steps")==0 and
          capture.get("geometry_audits_run") is False,"raw capture is not an unadmitted pre-audit record")
    _need(capture.get("base_candidate_sha256")==base_hash and
          Path(capture.get("base_candidate","")).resolve()==base_path,"raw capture base mismatch")
    _need(Path(capture.get("capture_helper_path","")).resolve()==RAW563.resolve() and
          capture.get("capture_helper_sha256")==sha(RAW563),"raw capture helper pin mismatch")
    _need(Path(capture.get("capture546_helper_path","")).resolve()==CAPTURE546.resolve() and
          capture.get("capture546_helper_sha256")==sha(CAPTURE546),"546 capture helper pin mismatch")
    receipt=capture.get("fit_receipt",{})
    _need(receipt.get("fit_entry_path")==str(FIT603) and receipt.get("fit_entry_sha256")==sha(FIT603),
          "captured fit entry pin mismatch")
    _need(receipt.get("delta_semantics")=="absolute_base_relative_millimeters",
          "raw direction is not an absolute base-relative delta")
    _need(receipt.get("goal_labels")==["face"],"partial direction objective labels differ")
    returns=capture.get("solver_returns",[])
    _need(capture.get("solver_return_count")==1 and capture.get("successful_delta_count")==1 and len(returns)==1,
          "raw capture does not contain exactly one emitted direction")
    ret=returns[0]
    candidate=Path(ret.get("candidate_path","")).resolve()
    direction_dir=(summary_path.parent/"raw-direction-001").resolve()
    direction_path=Path(ret.get("proposal_report_path","")).resolve()
    _need(candidate==direction_dir/"candidate.npz" and candidate.is_file(),"partial candidate path mismatch/missing")
    _need(direction_path==direction_dir/"report.json" and direction_path.is_file(),"partial direction receipt path mismatch/missing")
    _need(sha(candidate)==ret.get("candidate_sha256"),"partial candidate hash mismatch")
    _need(sha(direction_path)==ret.get("proposal_report_sha256"),"partial direction receipt hash mismatch")
    direction=readj(direction_path)
    _need(direction.get("schema")=="numi.human.raw_joint_lp_direction_patch563.v1" and
          direction.get("status")=="RAW_JOINT_LP_DIRECTION_FOR_EXISTING_549_PROJECTION",
          "raw partial direction status/schema mismatch")
    _need(direction.get("base_sha256")==base_hash and
          Path(direction.get("base_candidate","")).resolve()==base_path and
          direction.get("candidate_sha256")==sha(candidate),"partial direction base/candidate mismatch")
    _need(direction.get("raw_delta_semantics")=="absolute_base_relative_millimeters" and
          direction.get("raw_delta_units")=="millimeters","raw delta semantics/units mismatch")
    _need(direction.get("accepted_physical_steps")==0 and direction.get("native_admission") is False and
          direction.get("geometry_audits_run") is False and
          direction.get("candidate_arrays_copied_from_base_except_points_world_m") is True and
          direction.get("anchors_bitwise_unchanged") is True and
          direction.get("outside_scope_bitwise_unchanged") is True,
          "raw partial candidate invariants/limitations missing")
    _need(direction.get("solver_receipt",{}).get("partial_feasible_direction_emitted_as_offline_construction_only") is True,
          "partial direction is not labelled offline construction-only")
    _need(direction.get("solver_stage")=="joint_lp_partial_timeout_zero_fallback" and
          direction.get("solver_status")==1,"raw direction is not the recorded partial-timeout stage")
    _need(direction.get("source_objective_receipt",{}).get("fit_entry_sha256")==sha(FIT603),
          "partial direction source objective pin mismatch")

    solver=fit_report.get("solver",{});decomp=solver.get("component_decomposition_475",{})
    component_results=decomp.get("component_results",[])
    failures=[r for r in component_results if r.get("success") is not True]
    successes=[r for r in component_results if r.get("success") is True]
    _need(bool(component_results) and len(failures)==1 and len(successes)==len(component_results)-1,
          "partial fit must have exactly one failed component and all others complete")
    failed=failures[0];failed_component=failed.get("component")
    _need(failed_component==direction.get("solver_component") and
          "Time limit reached" in str(failed.get("message","")),"partial component failure does not match capture")
    sr=direction.get("solver_receipt",{});fb=sr.get("partial_fallback_receipt",{});cert=sr.get("feasibility_certificate",{})
    _need(sr.get("success") is False and sr.get("global_lp_converged") is False,
          "partial solver must remain explicitly non-converged")
    _need(fb.get("status")=="PARTIAL_FEASIBLE_COMPONENT_TIMEOUT_ZERO_FALLBACK" and
          fb.get("global_lp_converged") is False,"partial fallback status/convergence mismatch")
    for key in ("global_hard_rows_pass","global_no_regression_rows_pass","global_output_bounds_pass",
                "global_output_map_reconstruction_pass","global_soft_target_slack_rows_pass",
                "zeroed_component_rows_are_exact_zero","other_successful_component_deltas_retained"):
        _need(fb.get(key) is True,"partial fallback certificate missing true flag: "+key)
    for key in ("global_l1_optimality_claim","global_stage1_optimum_known","retained_gain_budget_claimed"):
        _need(fb.get(key) is False,"partial fallback incorrectly claims global optimality/gain")
    _need(fb.get("timed_out_components")==[failed_component] and
          fb.get("zeroed_components")==[failed_component] and
          fb.get("unknown_stage1_objective_components")==[failed_component],
          "timed-out component is not the exact-zero fallback component")
    _need(cert.get("pass") is True and type(cert.get("hard_rows")) is int and cert["hard_rows"]>0 and
          type(cert.get("output_bounds")) is int and cert["output_bounds"]>0 and
          type(cert.get("soft_target_rows")) is int and cert["soft_target_rows"]>0,
          "partial fallback row/bounds feasibility certificate incomplete")
    _need(solver.get("success") is False,"LP_NO_PROPOSAL solver unexpectedly reports success")

    snapshot_path=Path(direction.get("basis_snapshot_path","")).resolve()
    _need(snapshot_path==direction_dir/"basis-and-delta.npz" and snapshot_path.is_file(),
          "raw basis/delta snapshot path mismatch/missing")
    _need(sha(snapshot_path)==direction.get("basis_snapshot_sha256"),"raw basis/delta snapshot hash mismatch")
    with np.load(base_path,allow_pickle=False) as z:base_arrays={k:np.array(z[k],copy=True) for k in z.files}
    with np.load(candidate,allow_pickle=False) as z:target_arrays={k:np.array(z[k],copy=True) for k in z.files}
    _need(set(base_arrays)==set(target_arrays) and "points_world_m" in base_arrays,
          "partial target NPZ schema differs from base")
    base_points=base_arrays["points_world_m"];target_points=target_arrays["points_world_m"]
    _need(base_points.dtype==np.float32 and target_points.dtype==np.float32 and
          target_points.shape==base_points.shape and np.isfinite(base_points).all() and np.isfinite(target_points).all(),
          "partial F32 point arrays are nonfinite or mismatched")
    _need(all(_same_array(base_arrays[k],target_arrays[k]) for k in base_arrays if k!="points_world_m"),
          "partial candidate changed a nonpoint payload field")
    anchors=np.asarray(base_arrays["typed_coarse_anchor_output_rows_018"],dtype=np.int64).reshape(-1)
    _need(np.array_equal(base_points[anchors],target_points[anchors]),"partial candidate moved typed anchors")
    with np.load(snapshot_path,allow_pickle=False) as z:snap={k:np.array(z[k],copy=True) for k in z.files}
    for key in ("output_scope_rows","delta_mm","base_points_world_m_f32","basis_data","basis_indices",
                "basis_indptr","basis_shape","basis_coordinates","control_parent_rows"):
        _need(key in snap,"partial basis snapshot missing "+key)
    _need(_same_array(snap["base_points_world_m_f32"],base_points),"basis snapshot base points mismatch")
    scope=np.asarray(snap["output_scope_rows"]);delta=np.asarray(snap["delta_mm"])
    _need(scope.dtype.kind in "iu" and scope.ndim==1 and len(scope)==direction.get("output_scope_count") and
          len(np.unique(scope))==len(scope) and np.all((scope>=0)&(scope<len(base_points))),
          "partial output scope indices invalid")
    _need(delta.dtype==np.float64 and delta.shape==(3*len(scope),) and np.isfinite(delta).all(),
          "partial millimeter delta shape/dtype/finite check failed")
    _need(hashlib.sha256(delta.tobytes()).hexdigest()==direction.get("raw_delta_sha256"),
          "partial delta hash mismatch")
    bshape=np.asarray(snap["basis_shape"],dtype=np.int64).reshape(-1)
    bdata=np.asarray(snap["basis_data"]);bindices=np.asarray(snap["basis_indices"]);bindptr=np.asarray(snap["basis_indptr"])
    coords=np.asarray(snap["basis_coordinates"]);ncontrol=len(snap["control_parent_rows"])
    _need(bshape.shape==(2,) and tuple(bshape.tolist())==(len(scope),ncontrol) and
          bdata.ndim==1 and bindices.shape==bdata.shape and bindptr.shape==(len(scope)+1,) and
          coords.shape==(3*ncontrol,) and np.isfinite(bdata).all() and np.isfinite(coords).all() and
          bindptr[0]==0 and bindptr[-1]==len(bdata) and np.all(np.diff(bindptr)>=0) and
          np.all((bindices>=0)&(bindices<ncontrol)),"partial captured basis dimensions/finite check failed")
    _need(direction.get("basis_coordinate_reconstruction_residual_max_abs_mm")==0.0,
          "partial captured basis reconstruction receipt is not exact")
    rebuilt=base_points.copy()
    rebuilt[scope]=(base_points[scope].astype(np.float64)+delta.reshape(-1,3)/1000.0).astype(np.float32)
    _need(np.array_equal(rebuilt,target_points),"partial emitted F32 points do not reconstruct from captured delta")
    validator=loadmod("matched_fit583_partial_validator",MATCHED583)
    validator._validate_target_payload(base_path,candidate,allow_fit_receipt_updates=True)
    meta={"candidate_kind":"certified_partial","fit_status":"LP_NO_PROPOSAL",
          "partial_timeout":{"failed_component":failed_component,"failed_component_message":failed.get("message"),
              "completed_components":len(successes),"component_count":len(component_results),
              "fallback_status":fb.get("status"),"zeroed_components":fb.get("zeroed_components"),
              "global_lp_converged":False,"global_stage1_optimum_known":False,
              "global_l1_optimality_claim":False,"retained_gain_budget_claimed":False,
              "fit_report_sha256":sha(fit_dir/"report.json"),"capture_report":str(summary_path),
              "capture_report_sha256":sha(summary_path),"direction_report":str(direction_path),
              "direction_report_sha256":sha(direction_path),"basis_snapshot_sha256":sha(snapshot_path)}}
    return candidate,meta

def select_fit_candidate(fit_dir,base_path,fit_report,*,allow_certified_partial=False):
    fit_dir=Path(fit_dir);base_path=Path(base_path)
    if fit_report.get("status")=="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":
        if fit_report.get("inputs",{}).get("base_candidate_sha256")!=sha(base_path):
            raise ValueError("603 proposal base does not match unchanged round baseline")
        candidate=Path(fit_report.get("candidate_path",fit_dir/"candidate.npz"))
        if not candidate.is_file(): raise FileNotFoundError("603 complete candidate missing: "+str(candidate))
        return candidate,{"candidate_kind":"complete","fit_status":fit_report.get("status")}
    if fit_report.get("status")=="LP_NO_PROPOSAL":
        if not allow_certified_partial:
            raise ValueError("603 returned LP_NO_PROPOSAL; certified partial use requires --allow-certified-partial")
        return validate_certified_partial_candidate(fit_dir,base_path,fit_report)
    raise ValueError("603 did not emit a recognized complete or certified-partial proposal")

def check545(candidate,baseline,out,src):
    argv=build_check545_argv(candidate,baseline,out)
    stem=out.parent/(out.name+"-stage")
    audit=out/"audit-130/summary.json";pairs=out/"audit-130/surface-self-pairs.npy"
    try:
        stage(argv,stem,slot=False,sources=src+(candidate,baseline),timeout=1800)
    except RuntimeError:
        execution_path=stem.with_suffix(".execution.json")
        if not (execution_path.is_file() and (out/"report.json").is_file()): raise
        execution=readj(execution_path);report=readj(out/"report.json")
        if not recoverable_self_check(execution,report,audit,pairs,sha(candidate)): raise
        return report,audit,pairs
    execution=readj(stem.with_suffix(".execution.json"));report=readj(out/"report.json")
    if execution.get("returncode")!=0 or execution.get("source_pins_unchanged") is not True:
        raise RuntimeError("545 stage execution receipt is not a clean success")
    if report.get("status")!="COMPLETE_EXACT_CONSTRUCTION_CHECKS" or report.get("candidate_sha256")!=sha(candidate):
        raise ValueError("545 success report is incomplete or candidate-mismatched")
    return report,audit,pairs

def run(a):
    allow_partial=bool(getattr(a,"allow_certified_partial",False))
    owner_paths=owners();pins={str(p.resolve()):sha(p) for p in owner_paths}
    if sha(FROZEN_V4)!=FROZEN_V4_SHA: raise ValueError("frozen v4 source pin mismatch")
    state_path=a.state.resolve();s,roles,empty_pairs,query=strict_baseline(state_path)
    prior=a.prior_context.resolve();ancestry=a.ancestry_context.resolve()
    failed=(a.failed_candidate.resolve(),a.failed_audit130.resolve(),a.failed_pairs.resolve())
    paths_exist((prior,ancestry,*failed,FIT_TEMPLATE,MATCHED_TEMPLATE))
    seed_report=a.prior_seed_report.resolve() if a.prior_seed_report else None
    if seed_report: paths_exist((seed_report,))
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    current_state=state_path;last_failed=failed;results=[]
    try:
        for r in range(1,a.max_rounds+1):
            rd=out/f"round-{r:03d}";rd.mkdir()
            s,roles,empty_pairs,query=strict_baseline(current_state);base=roles["candidate"]
            failed_candidate,failed_audit,failed_pairs=last_failed
            depth_dir=rd/"depth-413"
            depth_argv=[PY,DEPTH,"--candidate",base,"--point-summary",roles["point_summary"],
                "--surface-report",roles["surface_report"],"--query-dir",query.parent,"--output-dir",depth_dir,
                "--expected-candidate-sha256",s["candidate_sha256"],
                "--expected-point-summary-sha256",s["point_summary_sha256"],
                "--expected-surface-report-sha256",s["surface_report_sha256"],
                "--expected-query-receipt-sha256",sha(query),"--expected-owner-sha256",DEPTH_OWNER_SHA]
            stage(depth_argv,rd/"depth-stage",slot=True,
                  sources=owner_paths+(current_state,base,roles["point_summary"],roles["surface_report"],query,
                     prior,ancestry,failed_candidate,failed_audit,failed_pairs),timeout=1800)
            depth_npz=depth_dir/"point-depth-rows.npz";depth_report=depth_dir/"365-compatible-report.json"
            paths_exist((depth_npz,depth_report))
            context_input=rd/"current-inputs.json"
            manifest={"state":str(current_state),"state_sha256":sha(current_state),
                "candidate":str(base),"candidate_sha256":sha(base),
                "point_summary":str(roles["point_summary"]),"point_summary_sha256":sha(roles["point_summary"]),
                "surface_report":str(roles["surface_report"]),"surface_report_sha256":sha(roles["surface_report"]),
                "query_receipt":str(query),"guard_context":str(prior),"source_parent_context":str(ancestry),
                "failed_candidate":str(failed_candidate),"failed_audit130":str(failed_audit),"failed_pairs":str(failed_pairs),
                "depth_rows_npz":str(depth_npz),"depth_rows_report":str(depth_report),
                "native_admission":False,"accepted_physical_steps":0}
            if seed_report: manifest["prior_seed_report"]=str(seed_report)
            writej(context_input,manifest)
            prep=rd/"prepared"
            cmd=[PY,PREP613,"--current-inputs",context_input,"--depth-rows-npz",depth_npz,
                 "--depth-rows-report",depth_report,"--fit-template",FIT_TEMPLATE,
                 "--matched-template",MATCHED_TEMPLATE,"--fit-entry",FIT603,"--out",prep]
            if seed_report: cmd += ["--prior-seed-report",seed_report]
            stage(cmd,rd/"prepare-stage",slot=False,
                  sources=owner_paths+(context_input,current_state,base,roles["point_summary"],
                    roles["surface_report"],query,depth_npz,depth_report,prior,ancestry,
                    failed_candidate,failed_audit,failed_pairs),timeout=3600)
            prepared=readj(prep/"prepared.json");context_path=prep/"constraints/constraint-context.json"
            if prepared.get("fit_not_run") is not True: raise ValueError("613 unexpectedly ran fit")
            prepared_pins=prepared.get("source_pins")
            if not isinstance(prepared_pins,dict) or not prepared_pins: raise ValueError("613 prepared source pins are missing")
            verify_pins(prepared_pins)
            prepared_sources=tuple(Path(p) for p in prepared_pins)
            paths_exist(prepared_sources)
            context=readj(context_path);seed_pairs=Path(context["seed_pairs_path"]);seed_report=Path(context["seed_report"])
            initial=json.loads((prep/"fit-argv.json").read_text())
            if not isinstance(initial,list): raise ValueError("fit argv is not JSON array")
            argv=set_scp1(initial,rd/"fit-000")
            writej(rd/"fit-argv-000.json",argv);attempts=[];last=None;cut_budget_stop=None
            for fi in range(MAX_EXTRA_CUT_FITS+1):
                argv_path=rd/f"fit-argv-{fi:03d}.json"
                if fi:
                    free=shutil.disk_usage(ROOT).free;action=additional_cut_action(free)
                    if action=="retain-last-fit":
                        cut_budget_stop={"fit_index":fi,"free_bytes":free,"stage":"before_cut_merge",
                                         "reason":"additional cut fit below 250 MiB; preserve completed proposal for retention"}
                        emit("cut-budget-stop",round=r,**cut_budget_stop);break
                    if action=="stop": disk_guard(free,MIN_RETENTION_FREE)
                    merge=rd/f"cut-merge-{fi:03d}"
                    cmd=[PY,MERGE609,"--base-candidate",base,"--current-seed-pairs",seed_pairs,
                         "--current-seed-report",seed_report,"--failed-candidate",last["candidate"],
                         "--failed-audit130",last["audit"],"--failed-self-pairs",last["pairs"],
                         "--fit-argv",rd/f"fit-argv-{fi-1:03d}.json","--output",merge]
                    stage(cmd,rd/f"cut-merge-stage-{fi:03d}",slot=False,
                          sources=owner_paths+(base,seed_pairs,seed_report,last["candidate"],last["audit"],last["pairs"],rd/f"fit-argv-{fi-1:03d}.json"),timeout=600)
                    plan=readj(merge/"fit-plan.json")
                    if plan.get("status")!="PREPARED_NOT_LAUNCHED": raise ValueError("609 merge did not prepare cut fit")
                    seed_pairs=merge/"self-seeds/merged-self-pair-seeds.npy";seed_report=merge/"self-seeds/seed-preflight.json"
                    argv=json.loads((merge/"fit-argv.json").read_text())
                    if not isinstance(argv,list) or Path(argv[argv.index("--base-candidate")+1])!=base: raise ValueError("609 changed the fixed round baseline")
                    if float(argv[argv.index("--time-limit-s")+1])!=90.: raise ValueError("609 changed fit budget")
                    writej(argv_path,argv)
                free=shutil.disk_usage(ROOT).free
                if fi==0:
                    disk_guard(free,MIN_FIT_FREE)
                else:
                    action=additional_cut_action(free)
                    if action=="retain-last-fit":
                        cut_budget_stop={"fit_index":fi,"free_bytes":free,"stage":"after_cut_merge_before_fit",
                                         "reason":"additional cut fit below 250 MiB; preserve completed proposal for retention"}
                        emit("cut-budget-stop",round=r,**cut_budget_stop);break
                    if action=="stop": disk_guard(free,MIN_RETENTION_FREE)
                verify_pins(prepared_pins)
                emit("fit-start",round=r,fit=fi,base_sha256=sha(base))
                stage(argv,rd/f"fit-stage-{fi:03d}",slot=True,
                      sources=owner_paths+prepared_sources+(current_state,base,context_path,seed_pairs,seed_report,argv_path),timeout=1800)
                fdir=Path(argv[argv.index("--output")+1]);fit_report_path=fdir/"report.json"
                fr=readj(fit_report_path)
                candidate,candidate_meta=select_fit_candidate(fdir,base,fr,allow_certified_partial=allow_partial)
                checkdir=rd/f"check-fit-{fi:03d}"
                check,audit,pairs=check545(candidate,current_state,checkdir,owner_paths)
                last={"candidate":candidate,"candidate_kind":candidate_meta["candidate_kind"],
                    "candidate_meta":candidate_meta,"audit":audit,"pairs":pairs,"report":check,"fit":fdir}
                attempts.append({"fit":fi,"candidate_sha256":sha(candidate),"fit_report_sha256":sha(fit_report_path),
                    "candidate_kind":candidate_meta["candidate_kind"],"partial_timeout":candidate_meta.get("partial_timeout"),
                    "check_status":check.get("status"),
                    "self_pairs":readj(audit).get("exact_PLC_surface_pairs") if audit.is_file() else None})
                emit("fit-checked",round=r,fit=fi,status=check.get("status"),self_pairs=attempts[-1]["self_pairs"])
                if check.get("status")!="FAILED": break
                if not is_exact_self_failure(check,audit,pairs,sha(candidate)):
                    raise ValueError("545 failed outside the complete exact self-pair gate")
                if fi==MAX_EXTRA_CUT_FITS: break
            if last is None: raise AssertionError("no 603 attempt")
            if last["report"].get("status")=="FAILED" and last["report"].get("error")!="exact self intersections":
                raise ValueError("last 603 candidate failed a non-self gate")
            disk_guard(shutil.disk_usage(ROOT).free,MIN_RETENTION_FREE)
            target=last["candidate"];ret=rd/"retention-582"
            emit("retention-start",round=r,target_sha256=sha(target))
            stage(build_retention582_argv(current_state,target,ret),rd/"retention-stage",slot=True,
                  sources=owner_paths+(current_state,base,target),timeout=1800)
            retained=ret/"candidate.npz";rr=readj(ret/"report.json")
            if rr.get("status")!="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED": raise ValueError("582 did not emit candidate")
            rcdir=rd/"retained-check";rcheck,raudit,rpairs=check545(retained,current_state,rcdir,owner_paths)
            if rcheck.get("status")!="COMPLETE_EXACT_CONSTRUCTION_CHECKS": raise ValueError("545 did not pass retained candidate")
            final=rd/"final"
            stage(build_finalizer546_argv(retained,current_state,raudit,rcdir/"audit-159/report.json",final),
                  rd/"finalize-stage",slot=False,
                  sources=owner_paths+(current_state,retained,raudit,rcdir/"audit-159/report.json"),timeout=600)
            fs=final/"state.json";fr=readj(final/"report.json")
            before=sum(int((b.get("result") or {}).get("count",0)) for b in readj(roles["surface_report"])["banks"])
            if not strict_improvement(fr,before): raise ValueError("strict finalizer did not reduce surface pairs")
            row={"status":"STRICT_PASS","baseline_state":str(current_state),"baseline_sha256":sha(base),
                 "allow_certified_partial":allow_partial,"selected_candidate_kind":last["candidate_kind"],
                 "selected_candidate_metadata":last["candidate_meta"],
                 "prepared":str(prep),"constraint_context":str(context_path),"next_seed_report":str(seed_report),
                 "fit_attempts":attempts,"cut_budget_stop":cut_budget_stop,
                 "retention_report":str(ret/"report.json"),"final_state":str(fs),
                 "final_state_sha256":sha(fs),"surface_pairs":fr["surface_pairs"],"contained_pairs":0,"self_pairs":0,
                 "native_admission":False,"accepted_physical_steps":0}
            writej(rd/"round-report.json",row);results.append(row)
            emit("strict-pass",round=r,state=str(fs),surface_pairs=fr["surface_pairs"])
            if r<a.max_rounds:
                current_state=fs;prior=context_path
                last_failed=(last["candidate"],last["audit"],last["pairs"])
                # For next round 613 maps this seed report against the prior guard candidate.
                # Keep the latest cut-merged seed report when one exists.
                if not all(Path(p).is_file() for p in last_failed): raise ValueError("next round seed witness incomplete")
        verify_pins(pins)
        return {"schema":"numi.human.connected_batch_619.v1","status":"MAX_ROUNDS_COMPLETE",
                "allow_certified_partial":allow_partial,"rounds":results,"source_pins":pins,
                "native_admission":False,"accepted_physical_steps":0}
    except Exception as e:
        verify_pins(pins)
        writej(out/"runner-report.json",{"schema":"numi.human.connected_batch_619.v1","status":"STOPPED",
            "allow_certified_partial":allow_partial,"completed_rounds":results,
            "error":type(e).__name__+": "+str(e),"source_pins":pins,
            "native_admission":False,"accepted_physical_steps":0})
        raise
    finally:
        if not (out/"runner-report.json").exists():
            writej(out/"runner-report.json",{"schema":"numi.human.connected_batch_619.v1",
                "status":"MAX_ROUNDS_COMPLETE","allow_certified_partial":allow_partial,
                "rounds":results,"source_pins":pins,
                "native_admission":False,"accepted_physical_steps":0})

def strict_improvement(report,before):
    return (report.get("status")=="COMPLETE_EXACT_CONSTRUCTION_CHECKS" and report.get("self_pairs")==0
            and report.get("contained_pairs")==0 and type(report.get("surface_pairs")) is int
            and report["surface_pairs"]<before)
def validate_round_count(value):
    if type(value) is not int or not 1<=value<=3: raise ValueError("max rounds must be in [1,3]")
    return value

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("state","prior-context","ancestry-context","failed-candidate","failed-audit130","failed-pairs","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--prior-seed-report",type=Path)
    p.add_argument("--max-rounds",type=int,default=1)
    p.add_argument("--allow-certified-partial",action="store_true",
                   help="opt in to a receipt-validated 564 zero-fallback direction after LP_NO_PROPOSAL")
    a=p.parse_args(argv)
    try: validate_round_count(a.max_rounds)
    except ValueError as e: p.error(str(e))
    result=run(a);emit("complete",status=result["status"],rounds=len(result["rounds"]));return 0
if __name__=="__main__": raise SystemExit(main())
