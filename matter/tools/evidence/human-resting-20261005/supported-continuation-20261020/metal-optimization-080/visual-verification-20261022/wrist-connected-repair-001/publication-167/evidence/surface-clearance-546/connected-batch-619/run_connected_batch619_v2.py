#!/usr/bin/env python3
"""Bounded repeated 613/603 continuation; construction evidence only, never launched by import."""
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
FIT_TEMPLATE=D/"connected-continuation-584/run-001/round-001/emitted-fit/fit-argv.json"
MATCHED_TEMPLATE=D/"connected-continuation-584/run-001/round-001/emitted-fit/matched-inputs.json"
CHECK545=ROOT/"D/contained-clearance-545/run_candidate_checks545.py"
MERGE609=D/"self-seed-merge-609/merge_self_seeds609_v2.py"
RETENTION582=D/"local-witness-retention-582/retain_local_witnesses582.py"
FINAL546=D/"finalize_existing_checks546.py"
MIN_FIT_FREE=250*1024*1024
MIN_RETENTION_FREE=165*1024*1024
MAX_EXTRA_CUT_FITS=2

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
       ROOT/"D/D/component-lp-decomposition-490/run_matched_component_fit_490.py",
       FIT603,FIT603_HELPER,CHECK545,MERGE609,RETENTION582,
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
def check545(candidate,baseline,out,src):
    argv=[PY,CHECK545,"--candidate",candidate,"--baseline-state",baseline,"--trust-mm","4","--output-dir",out]
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
    owner_paths=owners();pins={str(p.resolve()):sha(p) for p in owner_paths}
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
            writej(rd/"fit-argv-000.json",argv);attempts=[];last=None
            for fi in range(MAX_EXTRA_CUT_FITS+1):
                argv_path=rd/f"fit-argv-{fi:03d}.json"
                if fi:
                    disk_guard(shutil.disk_usage(ROOT).free,MIN_FIT_FREE)
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
                disk_guard(shutil.disk_usage(ROOT).free,MIN_FIT_FREE)
                verify_pins(prepared_pins)
                emit("fit-start",round=r,fit=fi,base_sha256=sha(base))
                stage(argv,rd/f"fit-stage-{fi:03d}",slot=True,
                      sources=owner_paths+prepared_sources+(current_state,base,context_path,seed_pairs,seed_report,argv_path),timeout=1800)
                fdir=Path(argv[argv.index("--output")+1]);candidate=fdir/"candidate.npz"
                fr=readj(fdir/"report.json")
                if fr.get("status")!="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED" or fr.get("inputs",{}).get("base_candidate_sha256")!=sha(base):
                    raise ValueError("603 did not emit proposal from unchanged round baseline")
                checkdir=rd/f"check-fit-{fi:03d}"
                check,audit,pairs=check545(candidate,current_state,checkdir,owner_paths)
                last={"candidate":candidate,"audit":audit,"pairs":pairs,"report":check,"fit":fdir}
                attempts.append({"fit":fi,"candidate_sha256":sha(candidate),"check_status":check.get("status"),
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
            stage([PY,RETENTION582,"--state",current_state,"--target",target,"--out",ret,
                   "--trust-mm","4","--max-edge-gradient",".5"],rd/"retention-stage",slot=True,
                  sources=owner_paths+(current_state,base,target),timeout=1800)
            retained=ret/"candidate.npz";rr=readj(ret/"report.json")
            if rr.get("status")!="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED": raise ValueError("582 did not emit candidate")
            rcdir=rd/"retained-check";rcheck,raudit,rpairs=check545(retained,current_state,rcdir,owner_paths)
            if rcheck.get("status")!="COMPLETE_EXACT_CONSTRUCTION_CHECKS": raise ValueError("545 did not pass retained candidate")
            final=rd/"final"
            stage([PY,FINAL546,"--candidate",retained,"--baseline-state",current_state,
                   "--point-summary",raudit,"--surface-report",rcdir/"audit-159/report.json","--out",final],
                  rd/"finalize-stage",slot=False,
                  sources=owner_paths+(current_state,retained,raudit,rcdir/"audit-159/report.json"),timeout=600)
            fs=final/"state.json";fr=readj(final/"report.json")
            before=sum(int((b.get("result") or {}).get("count",0)) for b in readj(roles["surface_report"])["banks"])
            if not strict_improvement(fr,before): raise ValueError("strict finalizer did not reduce surface pairs")
            row={"status":"STRICT_PASS","baseline_state":str(current_state),"baseline_sha256":sha(base),
                 "prepared":str(prep),"constraint_context":str(context_path),"next_seed_report":str(seed_report),
                 "fit_attempts":attempts,"retention_report":str(ret/"report.json"),"final_state":str(fs),
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
                "rounds":results,"source_pins":pins,"native_admission":False,"accepted_physical_steps":0}
    except Exception as e:
        verify_pins(pins)
        writej(out/"runner-report.json",{"schema":"numi.human.connected_batch_619.v1","status":"STOPPED",
            "completed_rounds":results,"error":type(e).__name__+": "+str(e),"source_pins":pins,
            "native_admission":False,"accepted_physical_steps":0})
        raise
    finally:
        if not (out/"runner-report.json").exists():
            writej(out/"runner-report.json",{"schema":"numi.human.connected_batch_619.v1",
                "status":"MAX_ROUNDS_COMPLETE","rounds":results,"source_pins":pins,
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
    a=p.parse_args(argv)
    try: validate_round_count(a.max_rounds)
    except ValueError as e: p.error(str(e))
    result=run(a);emit("complete",status=result["status"],rounds=len(result["rounds"]));return 0
if __name__=="__main__": raise SystemExit(main())
