#!/usr/bin/env python3
"""Guarded sequential source201 regional RAP tail-preservation helper command."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time, shlex
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SPEC_PATH = PACKAGE / "command-spec.json"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def fail(message):
    raise RuntimeError(message)

def process_table():
    raw = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,state=,command="],
        check=True, capture_output=True, text=True, timeout=10
    ).stdout
    rows = []
    for line in raw.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) == 4:
            try:
                rows.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
            except ValueError:
                pass
    return rows

def ancestors(rows):
    by_pid = {pid: ppid for pid, ppid, _state, _cmd in rows}
    keep, pid = {os.getpid()}, os.getpid()
    while pid in by_pid and by_pid[pid] not in keep and by_pid[pid] != 0:
        pid = by_pid[pid]
        keep.add(pid)
    return keep

def active_native_workers(rows):
    own = ancestors(rows)
    blocked, this_runner = [], str(Path(__file__).resolve())
    for pid, _ppid, state, command in rows:
        if pid in own or state.startswith("T") or this_runner in command:
            continue
        if "numi-matter-cardboard-" in command:
            blocked.append({"pid": pid, "state": state, "command": command[:500]})
        elif ("/analysis/native" in command or "/native/" in command) and (
            "runner.py --execute" in command or "run_native" in command
        ):
            blocked.append({"pid": pid, "state": state, "command": command[:500]})
    return blocked

def verify_bindings(spec, output_path=None):
    snapshot = Path(spec["source_snapshot"])
    frozen_path = Path(spec["frozen_source_manifest"]["path"])
    if sha256(frozen_path) != spec["frozen_source_manifest"]["sha256"]:
        fail("frozen-source.json hash mismatch")
    frozen = json.loads(frozen_path.read_text())
    expected_sources = spec["source_sha256"]
    if frozen.get("source_sha256") != expected_sources or len(expected_sources) != 817:
        fail("frozen source map/count mismatch")
    for rel, expected in expected_sources.items():
        path = snapshot / rel
        if not path.is_file() or sha256(path) != expected:
            fail("source hash mismatch: " + rel)
    for item in spec["bound_files"]:
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            fail("bound input mismatch: " + str(path))
    source_map = None
    for build in spec["builds"]:
        receipt_path = Path(build["receipt"]["path"])
        if sha256(receipt_path) != build["receipt"]["sha256"]:
            fail("build receipt hash mismatch: " + str(receipt_path))
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get("success"):
            fail("build did not succeed: " + str(receipt_path))
        current_map = receipt.get("source_sha256", {})
        if len(current_map) != 817 or current_map != expected_sources:
            fail("build source map mismatch: " + str(receipt_path))
        if source_map is None:
            source_map = current_map
        elif current_map != source_map:
            fail("O0/O3 source maps differ")
        if receipt.get("artifacts_sha256") != build["artifacts_sha256"]:
            fail("artifact receipt map mismatch: " + str(receipt_path))
        for artifact, expected in build["artifacts_sha256"].items():
            if not Path(artifact).is_file() or sha256(artifact) != expected:
                fail("artifact hash mismatch: " + artifact)
        for meta in build["metadata_files"]:
            p = Path(meta["path"])
            if not p.is_file() or sha256(p) != meta["sha256"]:
                fail("build metadata mismatch: " + str(p))
        compile_db_path = Path(build["compile_commands_path"])
        compile_db = json.loads(compile_db_path.read_text())
        matches = [entry for entry in compile_db
                   if Path(entry["file"]).name == build["checker_source_basename"]]
        if len(matches) != 1:
            fail("checker compile command is not unique: " + str(compile_db_path))
        tokens = shlex.split(matches[0]["command"])
        if build["host_build_type"] == "Debug":
            if "-g" not in tokens or any(token.startswith("-O") for token in tokens):
                fail("Debug checker compile command is not unoptimized")
        elif build["host_build_type"] == "Release":
            if "-O3" not in tokens:
                fail("Release checker compile command lacks -O3")
        else:
            fail("unsupported host optimization label")
    ctest_binding = spec["ctest_registration"]
    ctest_path = Path(ctest_binding["path"])
    if sha256(ctest_path) != ctest_binding["sha256"]:
        fail("CTest registration file hash mismatch")
    ctest = ctest_path.read_text()
    if ctest_binding["test_name"] not in ctest or 'RUN_SERIAL "TRUE"' not in ctest or 'TIMEOUT "120"' not in ctest:
        fail("CTest serial/timeout registration mismatch")
    out = Path(output_path or spec["output_path"])
    if os.path.lexists(out):
        fail("unique output path already exists: " + str(out))
    return {
        "source_count": len(expected_sources),
        "frozen_source_sha256": spec["frozen_source_manifest"]["sha256"],
        "release_artifact_sha256": spec["builds"][1]["artifacts_sha256"],
        "debug_artifact_sha256": spec["builds"][0]["artifacts_sha256"],
        "unique_output_absent": True,
    }

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    if sha256(Path(__file__)) != spec["runner_sha256"]:
        fail("runner script hash mismatch")
    binding = verify_bindings(spec)
    free_before = shutil.disk_usage(spec["workspace"]).free
    if free_before < spec["disk_guard"]["minimum_free_bytes"]:
        fail("disk guard refused: free bytes below threshold")
    blocked = active_native_workers(process_table())
    preflight = {
        "schema": "numi.cardboard.native534.preflight.v1",
        "mode": "execute" if args.execute else "preflight-only",
        "binding_check": binding,
        "free_bytes_before": free_before,
        "disk_guard_minimum_free_bytes": spec["disk_guard"]["minimum_free_bytes"],
        "worker_guard_passed": not blocked,
        "active_native_workers": blocked,
        "output_path_absent": True,
        "launched": False,
    }
    if args.preflight_only:
        preflight["status"] = "ready" if not blocked else "refused_active_native_worker"
        print(json.dumps(preflight, sort_keys=True, indent=2))
        return 0 if not blocked else 75
    if blocked:
        preflight["status"] = "refused_active_native_worker"
        print(json.dumps(preflight, sort_keys=True, indent=2), file=sys.stderr)
        return 75
    out = Path(spec["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.mkdir()
    except FileExistsError:
        fail("output path was concurrently claimed")
    child_env = {k:v for k,v in os.environ.items()
                 if not (k.startswith("NM_") or k.startswith("NUMI_"))}
    child_env["PATH"] = spec["path"]
    start, deadline = time.monotonic(), time.monotonic() + spec["timeout_seconds"]
    results = []
    for build in spec["builds"]:
        blocked = active_native_workers(process_table())
        if blocked:
            results.append({"optimization_label":build["label"],
                            "status":"refused_active_native_worker",
                            "active_native_workers":blocked})
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            results.append({"optimization_label":build["label"],
                            "status":"overall_timeout_exhausted"})
            break
        case = out / build["label"]
        case.mkdir()
        argv = [build["executable"], build["metallib"]]
        row = {
            "optimization_label":build["label"],
            "host_build_type":build["host_build_type"],
            "integrated_metal_optimization":build["integrated_metal_optimization"],
            "argv":argv,"timeout_seconds":remaining,"started_unix":time.time()
        }
        try:
            completed = subprocess.run(argv,cwd=spec["workspace"],env=child_env,
                capture_output=True,timeout=remaining,check=False)
            stdout,stderr=completed.stdout or b"",completed.stderr or b""
            (case/"stdout.bin").write_bytes(stdout)
            (case/"stderr.bin").write_bytes(stderr)
            row.update({"status":"completed","exit_code":completed.returncode,
                "stdout_sha256":hashlib.sha256(stdout).hexdigest(),
                "stderr_sha256":hashlib.sha256(stderr).hexdigest(),
                "stdout_bytes":len(stdout),"stderr_bytes":len(stderr),
                "finished_unix":time.time()})
        except subprocess.TimeoutExpired as exc:
            stdout=exc.stdout or b""; stderr=exc.stderr or b""
            if isinstance(stdout,str): stdout=stdout.encode()
            if isinstance(stderr,str): stderr=stderr.encode()
            (case/"stdout.bin").write_bytes(stdout)
            (case/"stderr.bin").write_bytes(stderr)
            row.update({"status":"timeout",
                "stdout_sha256":hashlib.sha256(stdout).hexdigest(),
                "stderr_sha256":hashlib.sha256(stderr).hexdigest(),
                "finished_unix":time.time()})
        results.append(row)
    post = verify_bindings(spec,output_path=str(PACKAGE/"unused-output-sentinel"))
    complete = len(results)==2 and all(r.get("status")=="completed" for r in results)
    parity = complete and results[0].get("exit_code")==results[1].get("exit_code")==0 and results[0]["stdout_sha256"]==results[1]["stdout_sha256"] and results[0]["stderr_sha256"]==results[1]["stderr_sha256"]
    free_after=shutil.disk_usage(spec["workspace"]).free
    report={**preflight,"mode":"executed","launched":True,"status":"pass" if parity else "failed",
        "results":results,"exact_output_and_exit_parity":bool(parity),
        "post_run_binding_check":post,"free_bytes_after":free_after,
        "disk_guard_reserve_satisfied":free_after>=spec["disk_guard"]["reserve_bytes"],
        "physical_transactions":0,"qualification_limit":spec["qualification_limit"]}
    (out/"run-receipt.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True,indent=2))
    return 0 if report["status"]=="pass" and report["disk_guard_reserve_satisfied"] else 1

if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print("source201 native534 command refused: "+str(exc),file=sys.stderr)
        raise SystemExit(2)
