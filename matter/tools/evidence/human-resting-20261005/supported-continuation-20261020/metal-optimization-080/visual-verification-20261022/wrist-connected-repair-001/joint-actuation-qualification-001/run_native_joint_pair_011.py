#!/usr/bin/env python3
"""Hash-pinned, matched selected-tendon core trials for native 011."""
import argparse, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019")
TEMPLATE = E / "results/009-knee-envelope-c10924a45efc-knee-flexors-8step-125us/baseline/invocation.json"
BINARY_DEFAULT = Path("/tmp/numi-human-native-joints-011")
BINARY_SHA_DEFAULT = "8cede7fe7a52c6e3945b0d04de7a4b2ebd88af5744074ec3f92df85e7b66ea9f"
BUILD_DIR = E.parent / "native-initial-joint-construction-011"
BUILD_RESULT = BUILD_DIR / "build-result.json"
COHORTS = {
    "knee-flexors": [342, 343, 367, 368, 382, 383, 407, 408],
    "knee-extensors": [365, 373, 374, 375, 405, 413, 414, 415],
    "achilles-plantarflexors": [348, 349, 369, 388, 389, 409],
}

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def base_argv(condition, output, cohort, steps, binary, seed_knees):
    d=json.loads(TEMPLATE.read_text())
    a=list(d["argv"])
    a[0]=str(binary); a[4]=str(output)
    a[a.index("--muscle-step-count")+1]=str(steps)
    a[a.index("--muscle-activation")+1]="0.0" if condition=="baseline" else "0.5"
    # Replace the template flexor set with the requested exact source IDs.
    out=[]; i=0
    while i<len(a):
        if a[i]=="--activated-source-muscle-index": i+=2; continue
        out.append(a[i]); i+=1
    a=out
    for idx in COHORTS[cohort]:
        a.extend(["--activated-source-muscle-index",str(idx)])
    if seed_knees:
        pos=a.index("--initial-reference-posture")+1
        a[pos:pos]=["--initial-reference-joint-q","106","0.05","--initial-reference-joint-q","120","0.05"]
    return a,d

def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")

def invoke(a, out, timeout):
    out.mkdir(parents=True,exist_ok=False)
    (out/"invocation.json").write_text(json.dumps({"argv":a,"cwd":str(ROOT)},indent=2)+"\n")
    t=time.perf_counter()
    try:
        with (out/"stdout.log").open("wb") as so, (out/"stderr.log").open("wb") as se:
            r=subprocess.run(a,cwd=ROOT,stdout=so,stderr=se,timeout=timeout,check=False)
        elapsed=time.perf_counter()-t
        receipt={"native_exit_code":r.returncode,"wall_seconds":elapsed,"argv":a,"stdout_sha256":sha(out/"stdout.log"),"stderr_sha256":sha(out/"stderr.log")}
    except subprocess.TimeoutExpired as e:
        elapsed=time.perf_counter()-t
        receipt={"native_exit_code":None,"timeout_seconds":timeout,"wall_seconds":elapsed,"argv":a,"timed_out":True}
    write_json(out/"execution.json",receipt)
    return receipt

def negative_guards(binary, binary_sha, outroot):
    a,_=base_argv("baseline",outroot/"invalid-placeholder","knee-flexors",1,binary,True)
    # Remove output path because parser must reject these conflicting options before execution.
    tests={
      "duplicate_initial_q": a+["--initial-reference-joint-q","106","0.06"],
      "root_pose_conflict": a+["--root-pose","0","0","0","0","0","0","1"],
    }
    rows=[]
    for name,argv in tests.items():
        t=time.perf_counter()
        r=subprocess.run(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20,check=False)
        txt=(r.stdout+r.stderr).decode("utf-8","replace")
        rows.append({"test":name,"exit_code":r.returncode,"wall_seconds":time.perf_counter()-t,"rejected":r.returncode!=0,"diagnostic":txt[-1200:],"argv":argv})
    report={"schema":"numi.human.joint-initial-q-admission-negative-checks.v1","binary":str(binary),"binary_sha256":binary_sha,"checks":rows,"all_rejected":all(x["rejected"] for x in rows)}
    write_json(outroot/"negative-cli-guards.json",report)
    if not report["all_rejected"]: raise RuntimeError("one or more conflicting CLI combinations were accepted")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binary",default=str(BINARY_DEFAULT)); ap.add_argument("--binary-sha256",default=BINARY_SHA_DEFAULT)
    ap.add_argument("--cohort",choices=sorted(COHORTS),required=True)
    ap.add_argument("--steps",type=int,default=8); ap.add_argument("--seed-knees",action="store_true")
    ap.add_argument("--run-name",required=True); ap.add_argument("--negative-guards",action="store_true")
    ap.add_argument("--timeout",type=float,default=90.0)
    ap.add_argument("--build-result",type=Path,default=BUILD_RESULT)
    ap.add_argument("--link-manifest",type=Path)
    ns=ap.parse_args()
    binary=Path(ns.binary).resolve(); actual=sha(binary)
    if actual!=ns.binary_sha256: raise SystemExit(f"binary hash mismatch: {actual}")
    if ns.seed_knees and ns.cohort=="achilles-plantarflexors": raise SystemExit("do not seed knees in Achilles pair")
    if ns.cohort.startswith("knee-") and not ns.seed_knees: raise SystemExit("knee cohort requires the declared 0.05 rad initial-only seed")
    outroot=E/"results"/ns.run_name
    if outroot.exists(): raise SystemExit(f"refusing to overwrite {outroot}")
    outroot.mkdir(parents=True)
    if ns.negative_guards: negative_guards(binary,actual,outroot)
    d=json.loads(TEMPLATE.read_text())
    build_result=json.loads(ns.build_result.read_text())
    if build_result.get("exit_code")!=0: raise SystemExit("build was not successful")
    build_meta={"build_result_path":str(ns.build_result),"build_result_sha256":sha(ns.build_result),"compiled_source_sha256":build_result["source_sha256"],"binary_sha256":actual}
    if ns.link_manifest:
        linked=json.loads(ns.link_manifest.read_text())
        if linked.get("exit_code")!=0 or linked["binary"]["sha256"]!=actual or linked.get("source_sha256")!=build_result["source_sha256"]:
            raise SystemExit("link/source/binary identity mismatch")
        build_meta["link_manifest_path"]=str(ns.link_manifest)
        build_meta["link_manifest_sha256"]=sha(ns.link_manifest)
    source_assets={k:v for k,v in d.get("input_hashes",{}).items()}
    for f,h in source_assets.items():
        if sha(f)!=h: raise SystemExit(f"input hash mismatch: {f}")
    rigid=d["candidate_rigid"]
    if sha(rigid["path"])!=rigid["sha256"]: raise SystemExit("candidate rigid hash changed")
    pair={"schema":"numi.human.native-joint-pair-011.v1","cohort":ns.cohort,"steps":ns.steps,"dt_s":0.000125,"binary":{"path":str(binary),"sha256":actual},"build_identity":build_meta,"source_assets":source_assets,"candidate_rigid":rigid,"initial_posture":"--initial-reference-posture; initial-only knee q106=q120=0.05 rad" if ns.seed_knees else "--initial-reference-posture only","activated_source_muscle_indices":COHORTS[ns.cohort],"conditions":{}}
    for condition in ("baseline","loaded"):
        output=outroot/condition
        argv,_=base_argv(condition,output,ns.cohort,ns.steps,binary,ns.seed_knees)
        rec=invoke(argv,output,ns.timeout)
        inv_path=output/"invocation.json"
        inv=json.loads(inv_path.read_text()); inv["binary_identity"]=pair["binary"]; inv["build_identity"]=build_meta; inv["condition"]=condition; inv["cohort"]=ns.cohort; inv["initial_posture"]=pair["initial_posture"]; inv_path.write_text(json.dumps(inv,indent=2,sort_keys=True)+"\n")
        pair["conditions"][condition]=rec
        if rec["native_exit_code"]!=0: break
    pair["complete"]=set(pair["conditions"])=={"baseline","loaded"} and all(v.get("native_exit_code")==0 for v in pair["conditions"].values())
    write_json(outroot/"pair.json",pair)
    print(json.dumps(pair,indent=2,sort_keys=True))
    if not pair["complete"]: raise SystemExit("pair incomplete or native trial failed; raw outputs retained")

if __name__=="__main__": main()
