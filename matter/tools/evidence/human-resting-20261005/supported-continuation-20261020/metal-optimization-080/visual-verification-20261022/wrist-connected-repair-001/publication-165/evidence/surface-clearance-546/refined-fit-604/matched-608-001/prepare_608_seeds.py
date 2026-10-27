#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, pathlib, sys
import numpy as np
ROOT=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069")
D=ROOT/"D/surface-clearance-546"
BASE=D/"connected-refinement-603/baseline-599-affine-filtered-002/candidate.npz"
CTX=D/"refined-fit-604/preparation-603-001/constraints/constraint-context.json"
FAILC=D/"refined-fit-604/matched-607-002/fit/candidate.npz"
FAIL130=D/"refined-fit-604/matched-607-002/direct-fit-full-check/audit-130/summary.json"
FAILPAIRS=D/"refined-fit-604/matched-607-002/direct-fit-full-check/audit-130/surface-self-pairs.npy"
OLDARGV=D/"refined-fit-604/matched-607-002/fit-argv.json"
REFRESH=D/"refresh_surface_constraints546.py"
PAIR=ROOT/"publication-101/evidence/204/source/fit/self_separation_026.py"
PACK=ROOT/"D/focused-clearance-541/constraints/refresh_guards_541.py"
EXPECTED={"baseline":"16a7f33f4b3d3ea7cf61bf55d01a1dca00fe9a32ee2cb7c4fa5f930531bb674e","refresh":"a9586a81d92f117040e774c980b6028168432386b2e9babd96a7b7f725175d09","pair":"c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3","pack":"293e4da49a10f07420d1da7c512823dcb182dca246e411c3775d18866b27a241"}
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def loadnpz(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in z.files}
def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def write(path,obj):pathlib.Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);a=ap.parse_args();out=pathlib.Path(a.output)
    script_path=out/"prepare_608_seeds.py"
    if out.exists() and any(p!=script_path for p in out.iterdir()):raise FileExistsError(f"output already has artifacts: {out}")
    input_paths=[BASE,CTX,FAILC,FAIL130,FAILPAIRS,OLDARGV,REFRESH,PAIR,PACK]
    pins={str(p):sha(p) for p in input_paths}
    for key,p in [("baseline",BASE),("refresh",REFRESH),("pair",PAIR),("pack",PACK)]:
        if pins[str(p)]!=EXPECTED[key]:raise ValueError(f"pinned {key} source changed: {pins[str(p)]}")
    ctx=json.loads(CTX.read_text())
    if ctx["candidate"]!=str(BASE) or ctx["candidate_sha256"]!=pins[str(BASE)]:raise ValueError("604 context does not bind exact 603 baseline")
    if ctx["seed_report_sha256"]!=sha(ctx["seed_report"]):raise ValueError("604 seed report hash mismatch")
    sr=json.loads(pathlib.Path(ctx["seed_report"]).read_text())
    if sr.get("candidate_sha256")!=pins[str(BASE)] or sr.get("output_pairs_sha256")!=sha(ctx["seed_pairs_path"]):raise ValueError("604 seed report mismatch")
    old=np.asarray(np.load(ctx["seed_pairs_path"],allow_pickle=False),dtype=np.int64)
    if old.shape!=(20417,2):raise ValueError(f"unexpected current seed shape {old.shape}")
    summary=json.loads(FAIL130.read_text())
    if summary.get("candidate_sha256")!=sha(FAILC) or summary.get("status")!="STOPPED_AT_SURFACE_GATE":raise ValueError("failed 607 exact audit is not bound to failed candidate or expected status")
    trial=np.asarray(np.load(FAILPAIRS,allow_pickle=False),dtype=np.int64)
    if trial.shape!=(3940,2) or int(summary.get("exact_PLC_surface_pairs",-1))!=len(trial):raise ValueError("607 exact self-pair count mismatch")
    base=loadnpz(BASE);failed=loadnpz(FAILC);F=np.asarray(base["source_O_faces"]);TF=np.asarray(failed["source_O_faces"])
    if F.shape!=TF.shape or not np.array_equal(F,TF):raise ValueError("607 topology differs; identity-only refresh is invalid")
    if sha(BASE)!=ctx["candidate_sha256"]:raise ValueError("baseline changed during preparation")
    refresh=mod("refresh546_seedreuse",REFRESH);packmod=mod("refresh541_pack",PACK)
    oldmap=refresh.map_seed_pairs_partial(F,F,old);newmap=refresh.map_seed_pairs_partial(TF,F,trial)
    if np.any(oldmap["source_status_code"]!=0) or len(oldmap["mapped_pairs"])!=len(old):raise ValueError("current seeds do not identity-map completely")
    if np.any(newmap["source_status_code"]!=0) or len(newmap["mapped_pairs"])!=len(trial):raise ValueError("failed 607 witnesses do not identity-map completely")
    oldset={tuple(map(int,x)) for x in oldmap["mapped_pairs"]};newset={tuple(map(int,x)) for x in newmap["mapped_pairs"]};overlap=oldset&newset
    if overlap:raise ValueError(f"new witnesses overlap existing seeds: {len(overlap)}")
    merged=np.unique(np.vstack((oldmap["mapped_pairs"],newmap["mapped_pairs"])),axis=0)
    if len(merged)!=len(old)+len(trial):raise ValueError("merged count does not equal old plus new")
    pair_owner=mod("pair_owner026_608",PAIR)
    rows,records=pair_owner.build_pair_rows(np.asarray(base["points_world_m"],np.float64)*1000.0,F,merged,margin_fraction=.25)
    rhs=max((float(r["lower"]) for r in rows),default=0.0)
    if not np.isfinite(rhs) or rhs>2e-8:raise ValueError(f"zero-step 026 certificate failed: {rhs}")
    packed=packmod.pack(rows);out.mkdir(parents=True,exist_ok=True);seed_dir=out/"self-seeds";seed_dir.mkdir()
    oldmap_path=seed_dir/"current-seed-identity-remap.npz";failmap_path=seed_dir/"failed-607-self-remap.npz"
    np.savez_compressed(oldmap_path,**oldmap["payload"]);np.savez_compressed(failmap_path,**newmap["payload"])
    merged_path=seed_dir/"merged-self-pair-seeds.npy";np.save(merged_path,merged)
    rows_path=seed_dir/"merged-self-pair-rows.npz";np.savez_compressed(rows_path,**packed)
    source_paths=[BASE,CTX,FAILC,FAIL130,FAILPAIRS,OLDARGV,REFRESH,PAIR,PACK,pathlib.Path(ctx["seed_pairs_path"]),pathlib.Path(ctx["seed_report"]),pathlib.Path(ctx["surface_report"]),pathlib.Path(ctx["guard_npz"]),pathlib.Path(ctx["guard_report"])]
    source_pins={str(p):sha(p) for p in source_paths}
    report={"schema":"numi.human.continuation-self-seeds.v1","status":"PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY","mapping_status":"PASS_PARTIAL_EXACT_FACE_REMAP_AND_ZERO_STEP_026_CERTIFICATE","candidate_path":str(BASE),"candidate_sha256":sha(BASE),"current_exact_self_pair_count":0,"zero_step_feasible":True,"zero_step_max_rhs_mm":rhs,"rebuilt_linear_row_count":len(rows),"merged_unique_pair_count":len(merged),"native_admission":False,"accepted_physical_steps":0,"construction_only":True,"current_seed_source":{"pairs_path":str(ctx["seed_pairs_path"]),"pairs_sha256":sha(ctx["seed_pairs_path"]),"source_pair_count":len(old),"mapped_unique_pair_count":len(oldmap["mapped_pairs"]),"dropped_pair_count":int(np.count_nonzero(oldmap["source_status_code"]!=0)),"mapping_payload_path":str(oldmap_path),"mapping_payload_sha256":sha(oldmap_path)},"failed_trial_source":{"candidate":str(FAILC),"candidate_sha256":sha(FAILC),"audit130_path":str(FAIL130),"audit130_sha256":sha(FAIL130),"pairs_path":str(FAILPAIRS),"pairs_sha256":sha(FAILPAIRS),"source_pair_count":len(trial),"mapped_unique_pair_count":len(newmap["mapped_pairs"]),"dropped_pair_count":int(np.count_nonzero(newmap["source_status_code"]!=0)),"mapping_payload_path":str(failmap_path),"mapping_payload_sha256":sha(failmap_path)},"deduplication":{"overlap_count":len(overlap),"merged_count_equals_sum":len(merged)==len(old)+len(trial)},"output_pairs_path":str(merged_path),"output_pairs_sha256":sha(merged_path),"output_rows_path":str(rows_path),"output_rows_sha256":sha(rows_path),"pair_owner_path":str(PAIR),"pair_owner_sha256":sha(PAIR),"source_pins":source_pins,"limitation":"Mapped construction-seed refresh and zero-step linear certificate only; no fit or exact post-fit audit was run."}
    report_path=seed_dir/"seed-preflight.json";write(report_path,report)
    argv=list(json.loads(OLDARGV.read_text()))
    def replace(flag,value):argv[argv.index(flag)+1]=str(value)
    replace("--self-pairs",merged_path);replace("--output",out/"fit");replace("--area-scp-iterations","1")
    fitargv_path=out/"fit-argv.json";write(fitargv_path,argv)
    fitplan={"status":"PREPARED_NOT_LAUNCHED","argv":argv,"fit_argv_path":str(fitargv_path),"fit_argv_sha256":sha(fitargv_path),"seed_preflight_path":str(report_path),"seed_preflight_sha256":sha(report_path),"base_candidate_path":str(BASE),"base_candidate_sha256":sha(BASE),"failed_trial_candidate_path":str(FAILC),"failed_trial_candidate_sha256":sha(FAILC),"self_pair_path":str(merged_path),"self_pair_sha256":sha(merged_path),"matched_inputs_json":"/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/refined-fit-604/matched-607-002/matched-inputs.json","changed_fit_arguments":["--self-pairs","--output","--area-scp-iterations"],"unchanged_fit_arguments":"All other pinned 607 arguments, including candidate, face topology, 606 point targets, 603 fixed guards, 490 matched inputs, source cap, trust, edge and area policies.","source_pins":source_pins,"native_admission":False,"accepted_physical_steps":0}
    plan_path=out/"fit-plan.json";write(plan_path,fitplan)
    print(json.dumps({"output":str(out),"seed_report":str(report_path),"seed_report_sha256":sha(report_path),"merged_pairs":len(merged),"mapped_failed_pairs":len(newmap["mapped_pairs"]),"new_overlap":len(overlap),"rows":len(rows),"zero_step_max_rhs_mm":rhs,"fit_argv":str(fitargv_path),"fit_argv_sha256":sha(fitargv_path),"fit_status":"PREPARED_NOT_LAUNCHED"},sort_keys=True))
if __name__=="__main__":main()
