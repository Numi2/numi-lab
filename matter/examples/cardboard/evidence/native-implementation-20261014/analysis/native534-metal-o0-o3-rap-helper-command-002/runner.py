#!/usr/bin/env python3
"""Guarded sequential Metal-O0/Metal-O3 source201 helper command."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time, shlex
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SPEC_PATH = PACKAGE / "command-spec.json"

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def fail(s): raise RuntimeError(s)

def ps_rows():
    raw=subprocess.run(["/bin/ps","-axo","pid=,ppid=,state=,command="],
        check=True,capture_output=True,text=True,timeout=10).stdout
    out=[]
    for line in raw.splitlines():
        a=line.strip().split(None,3)
        if len(a)==4:
            try: out.append((int(a[0]),int(a[1]),a[2],a[3]))
            except ValueError: pass
    return out

def ancestors(rows):
    pp={pid:parent for pid,parent,_state,_cmd in rows}
    result={os.getpid()}; cur=os.getpid()
    while cur in pp and pp[cur] not in result and pp[cur]!=0:
        cur=pp[cur]; result.add(cur)
    return result

def workers(rows):
    own=ancestors(rows); selfpath=str(Path(__file__).resolve()); bad=[]
    for pid,_parent,state,cmd in rows:
        if pid in own or state.startswith("T") or selfpath in cmd: continue
        if "numi-matter-cardboard-" in cmd:
            bad.append({"pid":pid,"state":state,"command":cmd[:500]})
        elif ("/analysis/native" in cmd or "/native/" in cmd) and (
            "runner.py --execute" in cmd or "run_native" in cmd):
            bad.append({"pid":pid,"state":state,"command":cmd[:500]})
    return bad

def check(spec, output_override=None):
    snap=Path(spec["source_snapshot"])
    frozen=spec["frozen_source_manifest"]
    if sha256(frozen["path"])!=frozen["sha256"]: fail("frozen source manifest hash mismatch")
    doc=json.loads(Path(frozen["path"]).read_text())
    sm=spec["source_sha256"]
    if doc.get("source_sha256")!=sm or len(sm)!=817: fail("817 source map mismatch")
    for rel,expected in sm.items():
        p=snap/rel
        if not p.is_file() or sha256(p)!=expected: fail("source mismatch: "+rel)
    for item in spec["bound_files"]:
        p=Path(item["path"])
        if not p.is_file() or sha256(p)!=item["sha256"]: fail("bound file mismatch: "+str(p))
    for build in spec["source_builds"]:
        receipt=build["receipt"]
        if sha256(receipt["path"])!=receipt["sha256"]: fail("source build receipt mismatch")
        d=json.loads(Path(receipt["path"]).read_text())
        if not d.get("success") or d.get("source_sha256")!=sm or len(d.get("source_sha256",{}))!=817:
            fail("source build source map mismatch")
        if d.get("artifacts_sha256")!=build["artifacts_sha256"]: fail("build artifact map mismatch")
        for path,want in build["artifacts_sha256"].items():
            if not Path(path).is_file() or sha256(path)!=want: fail("build artifact mismatch: "+path)
        for m in build["metadata_files"]:
            if not Path(m["path"]).is_file() or sha256(m["path"])!=m["sha256"]: fail("build metadata mismatch")
        comp=json.loads(Path(build["compile_commands_path"]).read_text())
        matches=[x for x in comp if Path(x["file"]).name==spec["checker_source_basename"]]
        if len(matches)!=1: fail("checker compile command missing/ambiguous")
        tokens=shlex.split(matches[0]["command"])
        if build["build_type"]=="Debug":
            if "-g" not in tokens or any(x.startswith("-O") for x in tokens): fail("debug host compile flags mismatch")
        elif build["build_type"]=="Release":
            if "-O3" not in tokens: fail("release host compile flags mismatch")
        else: fail("unknown host build type")
    ct=spec["ctest_registration"]
    if sha256(ct["path"])!=ct["sha256"]: fail("CTest metadata mismatch")
    txt=Path(ct["path"]).read_text()
    if ct["test_name"] not in txt or 'RUN_SERIAL "TRUE"' not in txt or 'TIMEOUT "120"' not in txt:
        fail("CTest serial/timeout contract mismatch")
    for run in spec["runs"]:
        if not Path(run["executable"]).is_file() or sha256(run["executable"])!=run["executable_sha256"]:
            fail("checker executable mismatch")
        if not Path(run["metallib"]).is_file() or sha256(run["metallib"])!=run["metallib_sha256"]:
            fail("Metal library mismatch: "+run["metallib"])
    if spec["runs"][0]["executable"]!=spec["runs"][1]["executable"]:
        fail("host binary differs between Metal optimization arms")
    out=Path(output_override or spec["output_path"])
    if os.path.lexists(out): fail("unique output path already exists")
    return {"source_count":817,"frozen_source_sha256":frozen["sha256"],
            "source_build_count":len(spec["source_builds"]),
            "metal_o0_sha256":spec["runs"][0]["metallib_sha256"],
            "metal_o3_sha256":spec["runs"][1]["metallib_sha256"],
            "same_host_binary":True,"output_absent":True}

def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight-only",action="store_true")
    g.add_argument("--execute",action="store_true")
    a=ap.parse_args()
    spec=json.loads(SPEC_PATH.read_text())
    if sha256(Path(__file__))!=spec["runner_sha256"]: fail("runner hash mismatch")
    binding=check(spec)
    free=shutil.disk_usage(spec["workspace"]).free
    if free<spec["disk_guard"]["minimum_free_bytes"]: fail("disk guard refused")
    blocked=workers(ps_rows())
    report={"schema":"numi.cardboard.native534.metal-optimization-preflight.v1",
      "binding_check":binding,"free_bytes_before":free,
      "disk_guard_minimum_free_bytes":spec["disk_guard"]["minimum_free_bytes"],
      "active_native_workers":blocked,"worker_guard_passed":not blocked,
      "output_path_absent":True,"launched":False}
    if a.preflight_only:
        report["status"]="ready" if not blocked else "refused_active_native_worker"
        print(json.dumps(report,sort_keys=True,indent=2)); return 0 if not blocked else 75
    if blocked:
        report["status"]="refused_active_native_worker"
        print(json.dumps(report,sort_keys=True,indent=2),file=sys.stderr); return 75
    out=Path(spec["output_path"]); out.parent.mkdir(parents=True,exist_ok=True)
    try: out.mkdir()
    except FileExistsError: fail("output path claimed concurrently")
    env={k:v for k,v in os.environ.items() if not (k.startswith("NM_") or k.startswith("NUMI_"))}
    env["PATH"]=spec["path"]
    deadline=time.monotonic()+spec["timeout_seconds"]; results=[]
    for run in spec["runs"]:
        blocked=workers(ps_rows())
        if blocked:
            results.append({"label":run["label"],"status":"refused_active_native_worker","workers":blocked}); break
        remaining=deadline-time.monotonic()
        if remaining<=0: results.append({"label":run["label"],"status":"overall_timeout"}); break
        case=out/run["label"]; case.mkdir()
        argv=[run["executable"],run["metallib"]]
        row={"label":run["label"],"metal_optimization":run["metal_optimization"],
             "host_build":"same source-bound Release helper for both arms","argv":argv,
             "timeout_seconds":remaining,"started_unix":time.time()}
        try:
            done=subprocess.run(argv,cwd=spec["workspace"],env=env,capture_output=True,timeout=remaining,check=False)
            stdout=done.stdout or b""; stderr=done.stderr or b""
            (case/"stdout.bin").write_bytes(stdout); (case/"stderr.bin").write_bytes(stderr)
            row.update({"status":"completed","exit_code":done.returncode,
              "stdout_sha256":hashlib.sha256(stdout).hexdigest(),
              "stderr_sha256":hashlib.sha256(stderr).hexdigest(),
              "stdout_bytes":len(stdout),"stderr_bytes":len(stderr),"finished_unix":time.time()})
        except subprocess.TimeoutExpired as e:
            stdout=e.stdout or b""; stderr=e.stderr or b""
            if isinstance(stdout,str): stdout=stdout.encode()
            if isinstance(stderr,str): stderr=stderr.encode()
            (case/"stdout.bin").write_bytes(stdout); (case/"stderr.bin").write_bytes(stderr)
            row.update({"status":"timeout","stdout_sha256":hashlib.sha256(stdout).hexdigest(),
              "stderr_sha256":hashlib.sha256(stderr).hexdigest(),"finished_unix":time.time()})
        results.append(row)
    post=check(spec,output_override=str(PACKAGE/"unused-output-sentinel"))
    complete=len(results)==2 and all(x.get("status")=="completed" for x in results)
    exact=complete and results[0].get("exit_code")==results[1].get("exit_code")==0 and results[0]["stdout_sha256"]==results[1]["stdout_sha256"] and results[0]["stderr_sha256"]==results[1]["stderr_sha256"]
    free_after=shutil.disk_usage(spec["workspace"]).free
    report.update({"status":"pass" if exact else "failed","launched":True,"runs":results,
      "exact_output_and_exit_parity":bool(exact),"post_run_binding_check":post,
      "free_bytes_after":free_after,"disk_guard_reserve_satisfied":free_after>=spec["disk_guard"]["reserve_bytes"],
      "physical_transactions":0,"qualification_limit":spec["qualification_limit"]})
    (out/"run-receipt.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True,indent=2))
    return 0 if report["status"]=="pass" and report["disk_guard_reserve_satisfied"] else 1

if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as e:
        print("source201 native534 refused: "+str(e),file=sys.stderr); raise SystemExit(2)
