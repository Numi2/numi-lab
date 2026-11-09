#!/usr/bin/env python3
"""Run hash-bound n64 repeats after the current isolated studies finish."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
exe = Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/numi_contact_fem_fp64_reference_n128")
manifest = e / "n128-final-inputs.sha256"
plan = e / "SPATIAL_N64_REBOUND_PLAN.md"
expected_binary = "7f8b311536d8bb7cfd8bf4eb053649da020d0c9a20dc621071284d9e207c2394"
expected_source = "019371ad99f9cfd0ff0a7c3b08e77d493f698f7b3f7d0ef528dd4695097edc1f"
receipt = e / "n64-rebound-005-supervisor.json"
free_limit = int(3.1 * 1024**3)
if receipt.exists():
    raise SystemExit("existing supervisor receipt")

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def command(pid):
    return subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                          capture_output=True, text=True).stdout.strip()

def active_workers(exclude=()):
    out = subprocess.check_output(["ps", "-Ao", "pid=,comm="], text=True)
    result = []
    for row in out.splitlines():
        parts = row.strip().split(None, 1)
        if len(parts) == 2 and int(parts[0]) not in exclude and Path(parts[1]).name.startswith(("numi_", "metalrobo_")):
            result.append({"pid": int(parts[0]), "executable": parts[1]})
    return result

start = time.monotonic()
analysis = root / "tests/contact/spatial_n64_rebound_005_analysis.py"
inputs = {str(p): digest(p) for p in (exe, manifest, plan, analysis, Path(__file__).resolve(),
          root / "tests/contact/large_spatial_reference.py",
          root / "tests/contact/spatial_n128_retry2_analysis.py",
          root / "tests/contact/spatial_n64_analysis.py",
          root / "tests/contact/spatial_refinement_analysis.py")}
record = {"input_sha256": inputs, "runs": [], "wait_limit_seconds": 96 * 3600,
          "worker_check_scope": "names beginning numi_ or metalrobo_; not all host activity"}

def finish(reason, code=1):
    record.update(reason=reason, wall_seconds=time.monotonic()-start)
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    raise SystemExit(code)

if inputs[str(exe)] != expected_binary or expected_source not in manifest.read_text():
    finish("frozen evaluator or source manifest mismatch")
# Preserve isolation while the control and conditional cost study run.
forbidden = {}
while True:
    live = False
    for pid, expected in forbidden.items():
        actual = command(pid)
        if actual and expected not in actual:
            finish("waiting process identity changed: " + str(pid))
        live |= bool(actual)
    if not live:
        break
    if time.monotonic() - start > record["wait_limit_seconds"]:
        finish("wait limit")
    time.sleep(15)
if any(digest(Path(p)) != h for p, h in inputs.items()):
    finish("frozen input changed while waiting")
for label, h in (("050", "0.0000005"), ("025", "0.00000025")):
    prefix = e / ("rk4-n64-rebound-005-h" + label)
    out, log, runmeta = [Path(str(prefix) + suffix) for suffix in (".json", ".log", "-execution.json")]
    if any(p.exists() for p in (out, log, runmeta)):
        finish("refusing to overwrite evidence")
    workers = active_workers()
    if workers:
        finish("recognized worker active before reference: " + json.dumps(workers))
    if shutil.disk_usage(e).free < free_limit:
        finish("free-space limit before reference")
    argv = [str(exe), "64", h, "0.01"]
    runstart = time.monotonic()
    peak = 0
    reason = None
    with out.open("x") as stdout, log.open("x") as stderr:
        proc = subprocess.Popen(["nice", "-n", "10", *argv], stdout=stdout, stderr=stderr, start_new_session=True)
        while proc.poll() is None:
            rss = subprocess.run(["ps", "-p", str(proc.pid), "-o", "rss="], capture_output=True, text=True).stdout.strip()
            if rss.isdigit():
                peak = max(peak, int(rss))
            conflicts = active_workers((proc.pid,))
            if conflicts:
                reason = "recognized competing Numi worker: " + json.dumps(conflicts)
            elif peak > 2 * 1024 * 1024:
                reason = "2 GiB resident-memory limit"
            elif shutil.disk_usage(e).free < free_limit:
                reason = "free-space limit"
            elif time.monotonic()-runstart > 2 * 3600:
                reason = "two-hour wall limit"
            if reason:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(5)
        code = proc.wait()
    run = {"command": argv, "scheduling": "nice 10", "binary_sha256": digest(exe),
           "source_sha256": expected_source, "manifest_sha256": digest(manifest),
           "returncode": code, "stopped_reason": reason,
           "wall_seconds_observational_only": time.monotonic()-runstart,
           "peak_rss_kib_sampled": peak, "output_sha256": digest(out), "log_sha256": digest(log),
           "limits": {"rss_kib": 2*1024*1024, "free_bytes": free_limit, "wall_seconds": 7200}}
    runmeta.write_text(json.dumps(run, indent=2) + "\n")
    record["runs"].append(run)
    print(json.dumps(run), flush=True)
    if code != 0 or reason is not None or run["binary_sha256"] != expected_binary:
        finish("reference incomplete or executable changed")
analysis_output = e / "spatial-n64-rebound-005-analysis.json"
analysis_error = e / "spatial-n64-rebound-005-analysis.log"
if analysis_output.exists() or analysis_error.exists():
    finish("refusing to overwrite analysis")
if any(digest(Path(path)) != value for path, value in inputs.items()):
    finish("frozen inputs changed during references")
with analysis_output.open("x") as out, analysis_error.open("x") as err:
    check = subprocess.run([sys.executable, str(analysis)], cwd=root, stdout=out, stderr=err)
record["analysis_returncode"] = check.returncode
record["analysis_sha256"] = digest(analysis_output)
finish("both hash-bound references completed and analyzed" if check.returncode == 0 else "reference analysis failed", check.returncode)
