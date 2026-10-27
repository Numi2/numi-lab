#!/usr/bin/env python3
"""Prepare a matched 613 continuation fit; never launches the fit."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
D=Path(__file__).resolve().parent.parent
PYTHON=Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
REFRESH=D/"refresh_surface_constraints546.py"
FACE_BUILDER=D/"source-current-targets-597/build_joint_outward_targets597_v4.py"
FACE_VERIFIER=D/"refined-fit-604/verify_joint_targets604.py"
FACE_COMPAT=D/"connected-fit-598/target-compatibility-001/compatibility_report598_v2.py"
JOINT_HELPER=D/"joint-refined-595/joint_refined595.py"
STAGE_HELPER=D/"continue_surface_round546.py"
FIT_ENTRY_DEFAULT=D/"component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"
FIT_TEMPLATE_DEFAULT=D/"connected-continuation-584/run-001/round-001/emitted-fit/fit-argv.json"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):
    v=json.loads(Path(p).read_text())
    if not isinstance(v,dict): raise ValueError("expected JSON object: "+str(p))
    return v
def writej(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n")
def loadmod(name,p):
    s=importlib.util.spec_from_file_location(name,p)
    if s is None or s.loader is None: raise ImportError(str(p))
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def loadnpz(p):
    import numpy as np
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def replace_cli_value(argv,flag,value):
    if argv.count(flag)!=1: raise ValueError(f"expected exactly one {flag}, got {argv.count(flag)}")
    i=argv.index(flag)
    if i+1>=len(argv) or argv[i+1].startswith("--"): raise ValueError(f"missing value after {flag}")
    argv[i+1]=str(value)
def set_fit_entry(argv,entry):
    entry=Path(entry)
    if not entry.is_file(): raise FileNotFoundError(entry)
    if len(argv)<2 or Path(argv[0])!=PYTHON: raise ValueError("fit argv interpreter differs from pinned workspace Python")
    argv[1]=str(entry)
def context_roles(guard_refresh,ancestry):
    gc=Path(guard_refresh["candidate_path"]);ac=Path(ancestry["candidate"])
    if not gc.is_file() or sha(gc)!=guard_refresh.get("candidate_sha256"): raise ValueError("guard context candidate pin mismatch")
    if not ac.is_file() or sha(ac)!=ancestry.get("candidate_sha256"): raise ValueError("ancestry context candidate pin mismatch")
    return gc,ac

def normalize_guard_context(path):
    value=readj(path)
    if "candidate_path" in value:
        return value, value
    # A 546 constraint-context is also accepted. Its refresh report is the
    # provenance receipt; its candidate/guard/seed paths are the actual roles.
    candidate=Path(value["candidate"]);refresh_path=Path(value["refresh_report"])
    report=readj(refresh_path)
    if value.get("refresh_report_sha256") and sha(refresh_path)!=value["refresh_report_sha256"]:
        raise ValueError("guard constraint-context refresh report hash mismatch")
    if report.get("candidate_sha256")!=sha(candidate) or value.get("candidate_sha256")!=sha(candidate):
        raise ValueError("guard constraint-context candidate mismatch")
    normalized=dict(report)
    normalized.update({"candidate_path":str(candidate),"candidate_sha256":sha(candidate),
        "audit159_path":value.get("surface_report",report.get("audit159_path")),
        "audit159_sha256":value.get("surface_report_sha256",report.get("audit159_sha256")),
        "fixed_guard_npz_path":value.get("guard_npz",report.get("fixed_guard_npz_path")),
        "fixed_guard_npz_sha256":sha(value.get("guard_npz",report.get("fixed_guard_npz_path"))),
        "fixed_guard_report_path":value.get("guard_report",report.get("fixed_guard_report_path")),
        "fixed_guard_report_sha256":sha(value.get("guard_report",report.get("fixed_guard_report_path"))),
        "seed_report_path":value.get("seed_report",report.get("seed_report_path")),
        "seed_pairs_path":value.get("seed_pairs_path",report.get("seed_pairs_path"))})
    return normalized,value
def current_inputs(p):
    manifest=readj(p);sp=Path(manifest["state"]);state=readj(sp)
    if sha(sp)!=manifest.get("state_sha256"): raise ValueError("state file hash mismatch")
    for role in ("candidate","point_summary","surface_report"):
        q=Path(manifest[role])
        if str(q)!=state.get(role) or sha(q)!=state.get(role+"_sha256"): raise ValueError("state role/hash mismatch: "+role)
    if state.get("native_admission") is not False or int(state.get("accepted_physical_steps", -1)) != 0:
        raise ValueError("preparation accepts construction evidence only")
    return manifest,state,sp

def validate_empty_baseline(state, summary, surface, depth_report, candidate_sha):
    if state.get("candidate_sha256") != candidate_sha or summary.get("candidate_sha256") != candidate_sha or surface.get("candidate_sha256") != candidate_sha:
        raise ValueError("zero-contained baseline reports are not candidate matched")
    if summary.get("complete_surface_audit") is not True or int(summary.get("exact_PLC_surface_pairs", -1)) != 0 or int(summary.get("inside_node_bank_pairs", -1)) != 0:
        raise ValueError("empty point objective requires complete zero-self, zero-contained baseline")
    if surface.get("complete") is not True or int(surface.get("bank_count", -1)) != 185 or int(surface.get("expected_bank_count", -1)) != 185:
        raise ValueError("empty point objective requires complete 185-bank surface report")
    if depth_report.get("candidate_sha256") != candidate_sha or int(depth_report.get("inside_row_count", -1)) != 0 or int(depth_report.get("outside_row_count", 0)) <= 0 or depth_report.get("complete_bank_accounting_verified") is not True or int(depth_report.get("bank_count", -1)) != 185:
        raise ValueError("empty point objective requires candidate-matched all-outside depth rows")
    return {"status":"PASS_EMPTY_POINT_OBJECTIVE_BASELINE","candidate_sha256":candidate_sha,"self_pairs":0,"contained_pairs":0,"exact_surface_banks":185,"native_admission":False,"accepted_physical_steps":0}
def validate_seed(report_path,prior_candidate):
    import numpy as np
    r=readj(report_path);pairs=Path(r.get("output_pairs_path",""))
    if r.get("candidate_sha256")!=sha(prior_candidate): raise ValueError("seed report not bound to prior guard candidate")
    if not pairs.is_file() or sha(pairs)!=r.get("output_pairs_sha256"): raise ValueError("prior seed pair hash mismatch")
    a=np.load(pairs,allow_pickle=False)
    if a.ndim!=2 or a.shape[1]!=2 or len(a)!=int(r.get("merged_unique_pair_count",-1)): raise ValueError("prior seed payload count/shape mismatch")
    return pairs,r
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current-inputs",type=Path,required=True)
    ap.add_argument("--point-target-npz",type=Path);ap.add_argument("--point-target-report",type=Path)
    ap.add_argument("--depth-rows-npz",type=Path);ap.add_argument("--depth-rows-report",type=Path)
    ap.add_argument("--prior-seed-report",type=Path)
    ap.add_argument("--fit-template",type=Path,default=FIT_TEMPLATE_DEFAULT)
    ap.add_argument("--fit-entry",type=Path,default=FIT_ENTRY_DEFAULT)
    ap.add_argument("--matched-template",type=Path)
    ap.add_argument("--out",type=Path,required=True);a=ap.parse_args(argv)
    m=loadmod("stage613",STAGE_HELPER);h=loadmod("joint595_613",JOINT_HELPER)
    manifest,state,state_path=current_inputs(a.current_inputs);base_dir=a.current_inputs.resolve().parent
    candidate=Path(state["candidate"]);summary=Path(state["point_summary"]);surface=Path(state["surface_report"])
    query=Path(manifest["query_receipt"])
    summary_obj=readj(summary);surface_obj=readj(surface)
    inside=int(summary_obj.get("inside_node_bank_pairs",-1))
    dnpz=a.depth_rows_npz or Path(manifest.get("depth_rows_npz",base_dir/"depth-413/point-depth-rows.npz"))
    drep=a.depth_rows_report or Path(manifest.get("depth_rows_report",base_dir/"depth-413/365-compatible-report.json"))
    for p in (dnpz,drep):
        if not p.is_file(): raise FileNotFoundError(p)
    pnpz=a.point_target_npz or Path(manifest.get("point_target_npz",base_dir/"point-targets/nearest-union-exits.npz"))
    prep=a.point_target_report or Path(manifest.get("point_target_report",base_dir/"point-targets/nearest-union-exits.emitted-f32-lineage.json"))
    if inside>0:
        for p in (pnpz,prep):
            if not p.is_file(): raise FileNotFoundError(p)
    grpath=Path(manifest["guard_context"]);guard,guard_input=normalize_guard_context(grpath)
    ancestry_path=Path(manifest["source_parent_context"]);ancestry=readj(ancestry_path)
    prior_candidate,ancestry_candidate=context_roles(guard,ancestry)
    if guard.get("fixed_guard_payload_contains_no_current_crossings") is not True: raise ValueError("prior guard context is not crossing-free")
    prior_surface=Path(guard["audit159_path"])
    if not prior_surface.is_file() or sha(prior_surface)!=guard.get("audit159_sha256"): raise ValueError("prior surface report pin mismatch")
    prior_gnpz=Path(guard["fixed_guard_npz_path"]);prior_grep=Path(guard["fixed_guard_report_path"])
    if sha(prior_gnpz)!=guard.get("fixed_guard_npz_sha256") or sha(prior_grep)!=guard.get("fixed_guard_report_sha256"): raise ValueError("prior guard payload pin mismatch")
    derived_seed=candidate.parent.parent/"self-seeds/seed-preflight.json"
    seed_default=manifest.get("prior_seed_report") or (derived_seed if derived_seed.is_file() else guard.get("seed_report_path"))
    if not seed_default: raise ValueError("no candidate-bound prior seed report was supplied or derived")
    seed_report=a.prior_seed_report or Path(seed_default)
    seed_pairs,seed=validate_seed(seed_report,prior_candidate)
    failed_candidate=Path(manifest["failed_candidate"]);failed_audit=Path(manifest["failed_audit130"]);failed_pairs=Path(manifest["failed_pairs"])
    fr=readj(failed_audit)
    import numpy as np
    fp=np.load(failed_pairs,allow_pickle=False)
    if fr.get("candidate_sha256")!=sha(failed_candidate) or len(fp)!=int(fr.get("exact_PLC_surface_pairs",-1)): raise ValueError("failed trial candidate/audit/pairs mismatch")
    depth_report=readj(drep)
    if depth_report.get("candidate_sha256")!=sha(candidate) or depth_report.get("complete_bank_accounting_verified") is not True or int(depth_report.get("bank_count",-1))!=185: raise ValueError("depth rows are not matched and complete")
    if inside>0:
        point_report=readj(prep)
        if point_report.get("inputs",{}).get("candidate_sha256")!=sha(candidate): raise ValueError("point target report candidate mismatch")
        validation=h.validate_joint_refined_inputs(state=state,point_summary=summary_obj,surface_report=surface_obj,
            arrays=loadnpz(candidate),candidate_sha256=sha(candidate),point_target_report=point_report,point_targets=loadnpz(pnpz),point_target_sha256=sha(pnpz),
            point_depth_report=depth_report,depth_npz=loadnpz(dnpz),depth_sha256=sha(dnpz),prior_context=ancestry,
            prior_candidate_sha256=sha(ancestry_candidate),ancestry_parent_sha256=sha(ancestry_candidate))
    else:
        validation=validate_empty_baseline(state,summary_obj,surface_obj,depth_report,sha(candidate))
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    if inside==0:
        pnpz,prep=m.write_empty_point_targets(out/"point-targets",candidate,summary,drep)
    cdir=out/"constraints"
    refresh_argv=[PYTHON,REFRESH,"--candidate",candidate,"--audit130",summary,"--audit159",surface,"--query-dir",query.parent,
        "--ancestry-parent-candidate",ancestry_candidate,"--prior-candidate",prior_candidate,"--prior-surface-report",prior_surface,
        "--prior-guard-candidate",prior_candidate,"--prior-guard-npz",prior_gnpz,"--prior-guard-report",prior_grep,
        "--prior-seed-pairs",seed_pairs,"--prior-seed-report",seed_report,"--failed-seed-candidate",failed_candidate,
        "--failed-seed-audit130",failed_audit,"--failed-seed-pairs",failed_pairs,"--output-dir",cdir,"--point-continuation"]
    refresh_sources=(Path(__file__),STAGE_HELPER,REFRESH,D/"refresh_guards_multi_parent_546.py",
        ROOT/"D/focused-clearance-541/constraints/refresh_guards_541.py",
        ROOT/"publication-101/evidence/204/source/fit/self_separation_026.py",
        ROOT/"publication-113/sources/connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py",
        ROOT/"publication-113/sources/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py",
        state_path,candidate,summary,surface,query,grpath,ancestry_path,prior_candidate,prior_surface,prior_gnpz,prior_grep,
        seed_report,seed_pairs,failed_candidate,failed_audit,failed_pairs,pnpz,prep,dnpz,drep)
    m.run_stage(refresh_argv,out/"refresh-stage",slot=True,source_paths=refresh_sources,timeout=1800)
    rrpath=cdir/"refresh-report.json";rr=readj(rrpath)
    if rr.get("candidate_sha256")!=sha(candidate): raise ValueError("refreshed guards not bound to current candidate")
    for path,key in ((rr["seed_pairs_path"],"seed_pairs_sha256"),(rr["fixed_guard_npz_path"],"fixed_guard_npz_sha256"),(rr["fixed_guard_report_path"],"fixed_guard_report_sha256")):
        if sha(path)!=rr.get(key): raise ValueError("refreshed output pin mismatch: "+key)
    fdir=out/"surface-targets"
    m.run_stage([PYTHON,FACE_BUILDER,"--candidate",candidate,"--surface-report",surface,"--query-receipt",query,"--output-dir",fdir,"--clearance-m",".00005","--direction-mode","source-outward"],
        out/"face-target-stage",slot=True,source_paths=(Path(__file__),FACE_BUILDER,FACE_BUILDER.with_name("angular_joint_direction597.py"),state_path,candidate,surface,query),timeout=1200)
    verify=out/"face-verification.json";compat=out/"face-fit-compatibility.json"
    m.run_stage([PYTHON,FACE_VERIFIER,"--folder",fdir,"--state",state_path,"--output",verify],out/"face-verification-stage",slot=False,
        source_paths=(FACE_VERIFIER,fdir/"targets.npz",fdir/"report.json",fdir/"target-records.jsonl"),timeout=180)
    m.run_stage([PYTHON,FACE_COMPAT,"--report",fdir/"report.json","--npz",fdir/"targets.npz","--records",fdir/"target-records.jsonl","--recovery",verify,"--out",compat],
        out/"face-compat-stage",slot=False,source_paths=(FACE_COMPAT,verify,fdir/"targets.npz",fdir/"report.json",fdir/"target-records.jsonl"),timeout=180)
    template=json.loads(a.fit_template.read_text())
    if not isinstance(template,list): raise ValueError("fit argv template must be a list")
    mt=a.matched_template or a.fit_template.with_name("matched-inputs.json");matched=out/"matched-inputs.json"
    m.matched_inputs_from_template(mt,matched,{"candidate":candidate,"audit159":surface,"face_target_npz":fdir/"targets.npz","point_target_npz":pnpz})
    fit=h.fit_argv_from_template(template,candidate=candidate,audit159=surface,point_target_npz=pnpz,point_target_report=prep,
        face_target_npz=fdir/"targets.npz",face_target_report=compat,point_depth_npz=dnpz,point_depth_report=drep,
        self_pairs=Path(rr["seed_pairs_path"]),fixed_guard_npz=Path(rr["fixed_guard_npz_path"]),fixed_guard_report=Path(rr["fixed_guard_report_path"]),
        matched_inputs=matched,output=out/"fit",trust_mm=4.0)
    set_fit_entry(fit,a.fit_entry)
    for flag in (x for x in fit if x.startswith("--")):
        if fit.count(flag)!=1: raise ValueError("duplicate fit flag: "+flag)
    m.write_json(out/"fit-argv.json",fit)
    pinfiles=(Path(__file__),STAGE_HELPER,JOINT_HELPER,REFRESH,FACE_BUILDER,FACE_VERIFIER,FACE_COMPAT,a.current_inputs,state_path,candidate,summary,surface,query,
        grpath,ancestry_path,prior_candidate,prior_surface,prior_gnpz,prior_grep,seed_report,seed_pairs,failed_candidate,failed_audit,failed_pairs,pnpz,prep,dnpz,drep,a.fit_template,a.fit_entry,a.fit_entry.with_name("component_scp_backtracking603.py"),mt,rrpath,
        cdir/"constraint-context.json",fdir/"targets.npz",fdir/"report.json",fdir/"target-records.jsonl",verify,compat,matched,out/"fit-argv.json")
    pins={str(p.resolve()):sha(p) for p in pinfiles}
    writej(out/"prepared.json",{"status":"MATCHED_613_POINT_AND_FACE_TARGETS_PREPARED_UNADMITTED",
        "current_state":str(state_path),"current_candidate":str(candidate),"current_candidate_sha256":sha(candidate),
        "guard_prior_context":str(grpath),"guard_prior_candidate":str(prior_candidate),"guard_prior_candidate_sha256":sha(prior_candidate),
        "source_ancestry_context":str(ancestry_path),"source_ancestry_candidate":str(ancestry_candidate),"source_ancestry_candidate_sha256":sha(ancestry_candidate),
        "prior_seed_report":str(seed_report),"prior_seed_pairs":str(seed_pairs),"prior_seed_pair_count":len(np.load(seed_pairs,allow_pickle=False)),
        "failed_seed_candidate":str(failed_candidate),"failed_seed_pair_count":len(fp),
        "point_target_report":str(prep),"point_target_npz":str(pnpz),"point_depth_report":str(drep),"point_depth_npz":str(dnpz),
        "refresh_report":str(rrpath),"refresh_report_sha256":sha(rrpath),"refresh_seed_pairs":rr["seed_pairs_path"],
        "refresh_guard_npz":rr["fixed_guard_npz_path"],"face_targets":str(fdir),"face_compatibility":str(compat),
        "matched_inputs":str(matched),"fit_argv":str(out/"fit-argv.json"),"fit_entry":str(a.fit_entry),"fit_entry_sha256":sha(a.fit_entry),"fit_not_run":True,
        "validation":validation,"source_pins":pins,"native_admission":False,"accepted_physical_steps":0})
    print(json.dumps({"status":"MATCHED_613_POINT_AND_FACE_TARGETS_PREPARED_UNADMITTED","out":str(out),"fit_not_run":True},sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())

