#!/usr/bin/env python3
"""Retry the preregistered BE ladder with losslessly compressed diagnostics.

Each producer remains exactly the frozen native program and physical setup.
Compression is streaming and outside the measured cost benchmark.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/midpoint-first-failure-20261109"
BUILD = Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-be-precision-005")
EXE = BUILD / "numi_plane_contact_probe_scaled"
SHADER = BUILD / "NumiMatter.metallib"
REFERENCE = ROOT / "docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n2-h025.json"
LADDER = [("00003125", "0.00000003125", 320000),
          ("0000625", "0.0000000625", 160000),
          ("000125", "0.000000125", 80000),
          ("00025", "0.00000025", 40000),
          ("0005", "0.0000005", 20000),
          ("001", "0.000001", 10000)]
# Compressed output is bounded by a guard 0.1 GiB above the concurrent
# reference guard; see the preregistered resource rationale.
LIMIT_FREE = int(3.1 * 1024**3)
LIMIT_RSS_KIB = 4 * 1024 * 1024
LIMIT_SECONDS = 24 * 3600

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def pump(pipe, path, errors, counts):
    h = hashlib.sha256()
    n = 0
    try:
        with path.open("xb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as out:
                while True:
                    chunk = pipe.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
        counts.update(uncompressed_bytes=n, uncompressed_sha256=h.hexdigest(),
                      compressed_bytes=path.stat().st_size, compressed_sha256=digest(path))
    except BaseException as error:
        errors.append(repr(error))
    finally:
        pipe.close()

def main():
    hashes = {str(p): digest(p) for p in (EXE, SHADER, REFERENCE)}
    if sys.argv[1:] == ["--audit"]:
        print(json.dumps({"inputs_sha256": hashes, "ladder": LADDER,
                          "free_bytes_limit": LIMIT_FREE}, indent=2))
        return 0
    if sys.argv[1:]:
        raise SystemExit("usage: run_be_compressed_ladder.py [--audit]")
    env = os.environ.copy()
    env["NM_MATTER_BE_VELOCITY_COMPENSATION"] = "1"
    for key in ("NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION",
                "NM_MATTER_MIDPOINT_STAGE_COMPENSATION",
                "NM_MATTER_MIDPOINT_PRECONTACT_POLISH",
                "NM_MATTER_MIDPOINT_DIAGNOSTIC_ROOT"):
        env.pop(key, None)
    for label, timestep, steps in LADDER:
        prefix = EVIDENCE / ("be-precision-006-n2-h" + label)
        paths = {suffix: Path(str(prefix) + suffix) for suffix in
                 (".json.gz", ".log.gz", "-execution.json", "-analysis.json")}
        if any(path.exists() for path in paths.values()):
            raise SystemExit("refusing to overwrite " + str(prefix))
        if shutil.disk_usage(EVIDENCE).free < LIMIT_FREE:
            raise SystemExit("free-space limit before " + str(prefix))
        command = [str(EXE), str(SHADER), timestep, str(steps), "2", "ipc_area",
                   "0.0002", "physical", "contact", "reference"]
        print("starting " + str(prefix), flush=True)
        start, reason, peak = time.monotonic(), None, 0
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        errors, out_counts, err_counts = [], {}, {}
        threads = [threading.Thread(target=pump, args=(process.stdout, paths[".json.gz"], errors, out_counts)),
                   threading.Thread(target=pump, args=(process.stderr, paths[".log.gz"], errors, err_counts))]
        for thread in threads:
            thread.start()
        while process.poll() is None:
            sample = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                                    capture_output=True, text=True)
            try:
                peak = max(peak, int(sample.stdout.strip()))
            except ValueError:
                pass
            if errors:
                reason = "diagnostic compression failure"
            elif peak > LIMIT_RSS_KIB:
                reason = "resident-memory limit"
            elif shutil.disk_usage(EVIDENCE).free < LIMIT_FREE:
                reason = "free-space limit"
            elif time.monotonic() - start > LIMIT_SECONDS:
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
        for thread in threads:
            thread.join()
        record = {"command": command, "input_sha256": hashes,
                  "flag": "NM_MATTER_BE_VELOCITY_COMPENSATION=1",
                  "returncode": code, "stopped_reason": reason,
                  "wall_seconds_observational_only": time.monotonic()-start,
                  "peak_rss_kib": peak, "free_bytes_after": shutil.disk_usage(EVIDENCE).free,
                  "limits": {"rss_kib": LIMIT_RSS_KIB, "free_bytes": LIMIT_FREE,
                             "wall_seconds": LIMIT_SECONDS},
                  "diagnostic_stdout": out_counts, "diagnostic_stderr": err_counts,
                  "compression_errors": errors, "all_gates_pass": False}
        try:
            with gzip.open(paths[".json.gz"], "rt") as f:
                state = json.load(f)
            check = subprocess.run(["python3", str(ROOT / "tests/contact/native_trace_analysis.py"),
                                    str(paths[".json.gz"]), str(REFERENCE)],
                                   capture_output=True, text=True)
            paths["-analysis.json"].write_text(check.stdout)
            record.update({"analysis_returncode": check.returncode, "analysis_stderr": check.stderr,
                           "all_gates_pass": code == 0 and reason is None and not errors
                           and check.returncode == 0 and state["accepted_steps"] == steps
                           and state["rejected_steps"] == 0
                           and state["independent_max_local_scaled_without_floor"] <= .005
                           and state["restore_replay"] and state["continuation_replay"]
                           and state["forced_rollback_exact"] and state["identity_mismatch_rejected"]})
        except (OSError, ValueError, KeyError) as error:
            record["analysis_error"] = repr(error)
        paths["-execution.json"].write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if not record["all_gates_pass"]:
            raise SystemExit("ladder stopped at first failure: " + str(prefix))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
