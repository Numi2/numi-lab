#!/usr/bin/env python3
"""Run the preregistered fixed-law BE accuracy ladder, with resource guards.

This is not a timing benchmark. The next timestep runs only if the previous
one completed its full rollout, physical checks, and exact replay. Stop at the
first failed or interrupted candidate; never relax an accuracy gate.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
evidence = root / "docs/evidence/midpoint-first-failure-20261109"
build = Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-be-precision-005")
exe = build / "numi_plane_contact_probe_scaled"
shader = build / "NumiMatter.metallib"
reference = root / "docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json"
ladder = [("0000625", "0.0000000625", 160000),
          ("000125", "0.000000125", 80000),
          ("00025", "0.00000025", 40000),
          ("0005", "0.0000005", 20000),
          ("001", "0.000001", 10000)]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (exe, shader, reference)}
if sys.argv[1:] == ["--audit"]:
    print(json.dumps({"inputs_sha256": hashes, "ladder": ladder}, indent=2))
    raise SystemExit(0)
if sys.argv[1:]:
    raise SystemExit("usage: run_be_qualified_ladder.py [--audit]")

previous = evidence / "be-precision-005-n2-h00003125-execution.json"
# Wait only for the already running, separately supervised fine control.
while not previous.exists():
    try:
        os.kill(94632, 0)
    except ProcessLookupError:
        raise SystemExit("fine-control supervisor exited without its execution record")
    time.sleep(30)
initial = json.loads(previous.read_text())
if not initial.get("all_gates_pass"):
    print("fine-control accuracy/replay gate failed; no coarser run started", flush=True)
    raise SystemExit(1)
if initial.get("input_sha256") != hashes:
    raise SystemExit("fine-control inputs do not match the frozen ladder inputs")

env = os.environ.copy()
env["NM_MATTER_BE_VELOCITY_COMPENSATION"] = "1"
for key in ("NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION", "NM_MATTER_MIDPOINT_STAGE_COMPENSATION"):
    env.pop(key, None)
for label, timestep, steps in ladder:
    prefix = evidence / ("be-precision-005-n2-h" + label)
    paths = {suffix: Path(str(prefix) + suffix) for suffix in
             (".json", ".log", "-execution.json", "-analysis.json")}
    if any(path.exists() for path in paths.values()):
        raise SystemExit("refusing to overwrite " + str(prefix))
    if shutil.disk_usage(evidence).free < int(3.2 * 1024**3):
        raise SystemExit("free-space limit before " + str(prefix))
    command = [str(exe), str(shader), timestep, str(steps), "2", "ipc_area",
               "0.0002", "physical", "contact", "reference"]
    start, reason, peak = time.monotonic(), None, 0
    print("starting " + str(prefix), flush=True)
    with paths[".json"].open("w") as output, paths[".log"].open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=log)
        while process.poll() is None:
            sample = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                                    capture_output=True, text=True)
            try:
                peak = max(peak, int(sample.stdout.strip()))
            except ValueError:
                pass
            if peak > 4 * 1024 * 1024:
                reason = "resident-memory limit"
            elif shutil.disk_usage(evidence).free < int(3.2 * 1024**3):
                reason = "free-space limit"
            elif time.monotonic() - start > 24 * 3600:
                reason = "24-hour limit"
            if reason:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(2)
        code = process.wait()
    record = {"command": command, "input_sha256": hashes,
              "flag": "NM_MATTER_BE_VELOCITY_COMPENSATION=1",
              "returncode": code, "stopped_reason": reason,
              "wall_seconds_observational_only": time.monotonic() - start,
              "peak_rss_kib": peak, "free_bytes_after": shutil.disk_usage(evidence).free,
              "limits": {"rss_kib": 4*1024*1024, "free_bytes": int(3.2*1024**3),
                         "wall_seconds": 24*3600}}
    try:
        state = json.loads(paths[".json"].read_text())
        check = subprocess.run(["python3", str(root / "tests/contact/native_trace_analysis.py"),
                                str(paths[".json"]), str(reference)], capture_output=True, text=True)
        paths["-analysis.json"].write_text(check.stdout)
        record.update({"analysis_returncode": check.returncode, "analysis_stderr": check.stderr,
                       "all_gates_pass": code == 0 and reason is None and check.returncode == 0
                       and state["accepted_steps"] == steps and state["rejected_steps"] == 0
                       and state["independent_max_local_scaled_without_floor"] <= .005
                       and state["restore_replay"] and state["continuation_replay"]
                       and state["forced_rollback_exact"] and state["identity_mismatch_rejected"]})
    except (ValueError, KeyError) as error:
        record.update({"analysis_error": repr(error), "all_gates_pass": False})
    paths["-execution.json"].write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if not record["all_gates_pass"]:
        raise SystemExit("ladder stopped at first failure: " + str(prefix))
print("all preregistered coarser candidates passed", flush=True)
