"""Explicitly matched 458/408 fit with opt-in 470 soft-target regression and 485 decomposition."""
from pathlib import Path
import hashlib,importlib.util,sys,json,os

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
W=D/"D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py"
H=D/"D/D/component-lp-decomposition-485/component_lp_decomposition_485.py"
S=D/"D/D/soft-target-nonregression-470/soft_target_nonregression_470.py"
O=D/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
FIXED= D/"D/D/fixed-bone-outward-plane-458/fixed_bone_outward_plane_458.py"
W_SHA="08cd0e2681bb5ae16a85a0384514bb30739f8d62a06ea2ab1bdfcd896be4ef93"
O_SHA="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
FIXED_SHA="056c8e60dad54f92a9c3ab69498dbfac0dc2c0152be272eadea2a58a3b6ea0e0"
SOFT_SHA="990e126932186e33c0a5363a6ac34fa967e211f4bc55977567582e2b89b8bb79"
HELPER_SHA="b8632e0734ff0163e854b9c0c5fbfe33c6d12c7b17cbb7935c5028c8df78ef74"
EXPECTED_CANDIDATE="742315227c1d8225aac9307af828251a73a98a9c40b4f59fe791aeef5da1ba0f"
EXPECTED_AUDIT="5d4e7995c08245a197e5e12df688fafa95473dfd27b34aa12456085d79b589eb"
EXPECTED_FACE="e9c05ed4bb92fcd3197e43988b3d904df70bb81e58fa6880145e616d200e9694"
EXPECTED_POINT="f43022866e3dd64e947554dece63ca17ca7d19dbd34332b3df622ec606c3becf"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    if spec is None or spec.loader is None: raise ImportError(f"cannot load pinned module {p}")
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def validate_source_pins():
    expected=((W,W_SHA),(O,O_SHA),(FIXED,FIXED_SHA),(S,SOFT_SHA),(H,HELPER_SHA))
    checked={}
    for path,digest in expected:
        if not path.is_file() or sha(path)!=digest:
            raise ValueError(f"485 frozen-source pin mismatch: {path}")
        checked[str(path)]=digest
    return checked

def validate_matched_inputs(candidate,audit,face_target,point_target):
    actual={"candidate":(Path(candidate),EXPECTED_CANDIDATE),
            "audit159":(Path(audit),EXPECTED_AUDIT),
            "face_target_npz":(Path(face_target),EXPECTED_FACE),
            "point_target_npz":(Path(point_target),EXPECTED_POINT)}
    checked={}
    for role,(path,digest) in actual.items():
        if not path.is_file() or sha(path)!=digest:
            raise ValueError(f"485 matched {role} input path/hash mismatch: {path}")
        checked[role]={"path":str(path.resolve()),"sha256":digest}
    report=json.loads(Path(audit).read_text())
    if report.get("candidate_sha256")!=EXPECTED_CANDIDATE:
        raise ValueError("485 matched 159 report does not bind to the fit candidate")
    return checked

def main():
    allow_flag="--allow-soft-target-regression-485"
    argv=list(sys.argv[1:])
    if argv.count(allow_flag)>1: raise ValueError("485 soft-regression opt-in flag may appear once")
    allow_soft=allow_flag in argv
    if allow_soft:
        argv.remove(allow_flag)
        sys.argv[:]=[sys.argv[0],*argv]
    pins=validate_source_pins()
    wrapper_path=Path(__file__).resolve()
    wrapper_sha_before=sha(wrapper_path)
    pins[str(wrapper_path)]=wrapper_sha_before
    w=load("matched458_for_485",W)
    soft=load("soft_target_noregression_470_for_485",S)
    helper=load("component_lp_decomposition_485_for_run",H)
    w.EXPECTED_CANDIDATE_SHA=EXPECTED_CANDIDATE
    w.EXPECTED_AUDIT_REPORT_SHA=EXPECTED_AUDIT
    w.EXPECTED_FACE_TARGET_SHA=EXPECTED_FACE
    w.EXPECTED_POINT_TARGET_SHA=EXPECTED_POINT
    if sha(H)!=HELPER_SHA: raise ValueError("485 component helper SHA mismatch")
    # Never allow a stale shell override to supersede the exact 458 inputs.
    os.environ["NUMI_FIXED_BONE_OUTWARD_CANDIDATE_SHA256_458"]=EXPECTED_CANDIDATE
    os.environ["NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458"]=EXPECTED_AUDIT
    if "NUMI_FIXED_BONE_OUTWARD_AUDIT159_PATH_458" not in os.environ:
        raise ValueError("exact matched audit path must be supplied by the launch environment")
    if os.environ.get("NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458")!=EXPECTED_AUDIT:
        raise ValueError("matched 159 audit environment pin mismatch")
    actual_preflight=validate_matched_inputs(
        w.arg_value("--base-candidate"),
        Path(os.environ["NUMI_FIXED_BONE_OUTWARD_AUDIT159_PATH_458"]),
        w.arg_value("--face-target-npz"),
        w.arg_value("--point-target-npz"))
    if actual_preflight["candidate"]["sha256"]!=EXPECTED_CANDIDATE:
        raise ValueError("matched candidate preflight mismatch")
    max_wall=float(os.environ.get("NUMI_COMPONENT_LP_MAX_TOTAL_WALL_S_485","120"))
    if not (0 < max_wall <= 120): raise ValueError("485 total solver budget must be in (0,120]")
    original=w.install_objective
    installs={}
    def install(owner,objective,crossing):
        if allow_soft:
            installs["470"]=soft.install_solver(owner)
        installs["485"]=helper.install(owner,allow_soft_target_regression=allow_soft,max_total_wall_s=max_wall)
        installs["owner"]=owner
        return original(owner,objective,crossing)
    w.install_objective=install
    w.main()
    matched_paths={
      "candidate":w.arg_value("--base-candidate"),
      "audit159":Path(os.environ["NUMI_FIXED_BONE_OUTWARD_AUDIT159_PATH_458"]),
      "face_target_npz":w.arg_value("--face-target-npz"),
      "point_target_npz":w.arg_value("--point-target-npz"),
    }
    matched_shas={"candidate":EXPECTED_CANDIDATE,"audit159":EXPECTED_AUDIT,
                  "face_target_npz":EXPECTED_FACE,"point_target_npz":EXPECTED_POINT}
    for role,path in matched_paths.items():
        if not path.is_file() or sha(path)!=matched_shas[role]:
            raise RuntimeError(f"matched {role} input changed during run")
    out=w.arg_value("--output")/"report.json"
    report=json.loads(out.read_text())
    if "owner" not in installs: raise RuntimeError("458 did not install the wrapped 408 solver")
    report["matched_component_fit_485"]={
      "classification":"construction-only sparse component LP using frozen 408/458 owners",
      "soft_target_regression_opt_in":allow_soft,
      "soft_target_regression_marker_required":"__soft_target_no_regression_470__ is true on installed solver" if allow_soft else None,
      "soft_target_scope":"Only C@delta >= 0 is omitted when explicitly enabled; target slack inequalities/bounds, physical A rows, output bounds, and global weighted-slack budget remain certified.",
      "soft_target_progress_may_be_negative":allow_soft,
      "negative_progress_count_and_minimum_are_reported_by_solver":True,
      "matched_inputs_preflight":actual_preflight,
      "component_solver_installation":installs["485"],
      "soft_target_solver_installation":installs.get("470"),
      "component_helper_path":str(H),"component_helper_sha256":sha(H),
      "soft_target_helper_path":str(S),"soft_target_helper_sha256":sha(S),
      "wrapper_path":str(Path(__file__).resolve()),"wrapper_sha256":sha(__file__),
      "frozen_458_sha256":sha(W),"frozen_408_sha256":sha(O),
      "matched_input_pins_after_run":{role:{"path":str(path),"sha256":sha(path)} for role,path in matched_paths.items()},
      "global_stage2_l1_optimality_claim":False,"native_admission":False,"accepted_physical_steps":0,
    }
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    after={str(p):sha(p) for p in (W,O,FIXED,S,H,wrapper_path)}
    if any(after[k]!=v for k,v in pins.items()): raise RuntimeError("frozen source changed during fit")
    report=json.loads(out.read_text());report["matched_component_fit_485"]["source_pins_after_run"]=after
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

if __name__=="__main__": main()
