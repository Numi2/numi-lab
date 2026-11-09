#!/usr/bin/env python3
"""Conditionally run the frozen bounded-memory reference after other studies."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from large_spatial_reference import read_reference, compare

root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
exe = Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-compact-005/numi_contact_fem_fp64_reference_compact")
plan = e / "LARGE_SPATIAL_CONTINUATION_PLAN.md"
receipt = e / "large-spatial-continuation-006-supervisor.json"
source = root / "tests/contact/fem_plane_fp64_reference.cpp"
files = [Path(__file__).resolve(), root / "tests/contact/large_spatial_reference.py", plan,
         source, exe, e / "compact-reference-n256-rk4-capacity-002-execution.json",
         e / "compact-reference-retained-n32-parity-001-analysis.json"]
expected_binary = "8b1cc61827aa63660320011e6f6e9f3c3deaab9fa838aa1706d428aa1bc177ac"
expected_source = "f32ce000b8f5001c664684725a3ed055a75af0cbc9b17634fbddbf752e3b0bc1"
expected_included = "6d75d723f79ba704788e506e81e5a214ebf105079d9cb0d4ef78cfe842d1a237"
free_limit = int(3.1 * 1024**3)
rss_limit = 6 * 1024 * 1024
run_limit = 7 * 24 * 3600
wait_limit = 96 * 3600


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(pid):
    return subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout.strip()


def workers(exclude=()):
    result = []
    for row in subprocess.check_output(["ps", "-Ao", "pid=,comm="], text=True).splitlines():
        parts = row.strip().split(None, 1)
        if len(parts) == 2 and int(parts[0]) not in exclude and Path(parts[1]).name.startswith(("numi_", "metalrobo_")):
            result.append({"pid": int(parts[0]), "executable": parts[1]})
    return result


def pump(source_pipe, destination, errors):
    try:
        with source_pipe, gzip.open(destination, "xb", compresslevel=6) as out:
            shutil.copyfileobj(source_pipe, out, length=1024 * 1024)
    except Exception as exc:
        errors.append(repr(exc))


def run_reference(n, h, label):
    prefix = e / ("rk4-compact-large-006-" + label)
    out = Path(str(prefix) + ".json.gz")
    log = Path(str(prefix) + ".log")
    meta = Path(str(prefix) + "-execution.json")
    if any(p.exists() for p in (out, log, meta)):
        raise RuntimeError("refusing to overwrite " + str(prefix))
    active = workers()
    if active or shutil.disk_usage(e).free < free_limit:
        raise RuntimeError("worker or storage guard before run: " + json.dumps(active))
    argv = [str(exe), str(n), h, "0.01"]
    start = time.monotonic()
    peak = 0
    reason = None
    errors = []
    with log.open("x") as err:
        proc = subprocess.Popen(["nice", "-n", "10", *argv], stdout=subprocess.PIPE, stderr=err, start_new_session=True)
        thread = threading.Thread(target=pump, args=(proc.stdout, out, errors), daemon=True)
        thread.start()
        while proc.poll() is None:
            rss = subprocess.run(["ps", "-p", str(proc.pid), "-o", "rss="], capture_output=True, text=True).stdout.strip()
            if rss.isdigit():
                peak = max(peak, int(rss))
            active = workers((proc.pid,))
            if errors:
                reason = "compression failed: " + repr(errors)
            elif active:
                reason = "recognized competing Numi worker: " + json.dumps(active)
            elif peak > rss_limit:
                reason = "6 GiB resident-memory limit"
            elif shutil.disk_usage(e).free < free_limit:
                reason = "3.1 GiB free-space limit"
            elif time.monotonic() - start > run_limit:
                reason = "seven-day per-run limit"
            if reason:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            time.sleep(5)
        code = proc.wait()
        thread.join(timeout=60)
    if thread.is_alive():
        errors.append("compression thread did not finish")
    record = {"command": argv, "scheduling": "nice 10", "binary_sha256": digest(exe),
              "source_git_revision": "9a1b1f78e", "source_sha256": expected_source,
              "included_source_sha256": digest(source), "returncode": code, "stopped_reason": reason,
              "compression_errors": errors, "peak_rss_kib_sampled": peak,
              "wall_seconds_observational_only": time.monotonic() - start,
              "output_sha256": digest(out) if out.exists() and not thread.is_alive() else None, "log_sha256": digest(log),
              "limits": {"rss_kib": rss_limit, "free_bytes": free_limit, "wall_seconds": run_limit}}
    meta.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if code != 0 or reason is not None or errors or record["binary_sha256"] != expected_binary:
        raise RuntimeError("reference did not complete")
    return out


def main():
    if receipt.exists():
        raise SystemExit("existing supervisor receipt")
    start = time.monotonic()
    hashes = {str(path): digest(path) for path in files}
    record = {"input_sha256": hashes, "runs": [], "wait_limit_seconds": wait_limit,
              "worker_check_scope": "process names beginning numi_ or metalrobo_; not all host activity"}
    def finish(reason, status=1):
        record.update(reason=reason, elapsed_seconds=time.monotonic()-start)
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        return status
    old_source = subprocess.check_output(["git", "show", "9a1b1f78e:tests/contact/fem_plane_fp64_reference_compact.cpp"], cwd=root)
    capacity = json.loads((e / "compact-reference-n256-rk4-capacity-002-execution.json").read_text())
    parity = json.loads((e / "compact-reference-retained-n32-parity-001-analysis.json").read_text())
    if (hashes[str(exe)] != expected_binary or hashes[str(source)] != expected_included or
        hashlib.sha256(old_source).hexdigest() != expected_source or
        capacity.get("binary_sha256") != expected_binary or not capacity.get("basic_invariants_pass") or
        not parity.get("passes") or parity.get("compact_binary_sha256") != expected_binary or
        parity.get("compact_source_sha256") != expected_source or parity.get("included_source_sha256") != expected_included or
        capacity.get("returncode") != 0 or capacity.get("stopped_reason") is not None or
        capacity.get("command") != [str(exe), "--rk4-capacity-check", "256"] or
        capacity.get("peak_rss_bytes_self", rss_limit * 1024 + 1) > rss_limit * 1024):
        return finish("frozen reference provenance or prior prerequisites failed")
    # The retry coordinator invokes this only after the n64 process and cost
    # study have stopped; the receipt and worker guards below remain decisive.
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        return finish("frozen input changed while waiting")
    def ensure_inputs():
        if any(digest(Path(p)) != h for p, h in hashes.items()):
            raise RuntimeError("frozen input changed during continuation")
    try:
        prior_path = e / "spatial-n64-rebound-006-analysis.json"
        prior = json.loads(prior_path.read_text())
        prior_receipt = json.loads((e / "n64-rebound-006-supervisor.json").read_text())
        if (prior_receipt.get("reason") != "both hash-bound references completed and analyzed" or
            prior_receipt.get("analysis_sha256") != digest(prior_path) or not prior.get("spatial_test_valid")):
            return finish("earlier mesh time refinement or provenance did not qualify")
        if prior.get("spatial_qualified") is True:
            return finish("earlier spatial comparison already qualified", 0)
        if prior.get("spatial_qualified") is not False or prior.get("n64_to_n128", {}).get("passes") is not False:
            return finish("earlier spatial decision missing or inconsistent")
        earlier_runs = [json.loads((e / ("rk4-n64-rebound-" + iteration + "-h" + label + "-execution.json")).read_text())
                        for iteration, label in (("005", "050"), ("006", "025"))]
        if len(earlier_runs) != 2:
            return finish("n64 execution records missing")
        for label, h, run in zip(("050", "025"), ("0.0000005", "0.00000025"), earlier_runs):
            old64 = e / ("rk4-n64-rebound-" + ("005" if label == "050" else "006") + "-h" + label + ".json")
            if (run.get("binary_sha256") != "7f8b311536d8bb7cfd8bf4eb053649da020d0c9a20dc621071284d9e207c2394" or
                run.get("returncode") != 0 or run.get("stopped_reason") is not None or
                run.get("source_sha256") != "019371ad99f9cfd0ff0a7c3b08e77d493f698f7b3f7d0ef528dd4695097edc1f" or
                run.get("output_sha256") != digest(old64) or run.get("command", [])[1:] != ["64", h, "0.01"]):
                return finish("n64 reference output or provenance changed")
        record["n64_analysis_sha256"] = digest(prior_path)
        old128 = e / "rk4-n128-retry1-h050.json"
        oldmeta = json.loads((e / "rk4-n128-retry1-h050-execution.json").read_text())
        if oldmeta.get("returncode") != 0 or oldmeta.get("stopped_reason") is not None or oldmeta.get("binary_sha256") != "7f8b311536d8bb7cfd8bf4eb053649da020d0c9a20dc621071284d9e207c2394":
            return finish("n128 parity reference provenance failed")
        n128_analysis = json.loads((e / "spatial-n128-retry2-analysis.json").read_text())
        fine128 = e / "rk4-n128-retry2-h025.json"
        fine_meta = json.loads((e / "rk4-n128-retry2-h025-execution.json").read_text())
        if (n128_analysis.get("inputs_sha256", {}).get("h050") != digest(old128) or
            n128_analysis.get("inputs_sha256", {}).get("h025") != digest(fine128) or
            not n128_analysis.get("n128_temporal", {}).get("passes") or
            fine_meta.get("returncode") != 0 or fine_meta.get("stopped_reason") is not None or
            fine_meta.get("binary_sha256") != "7f8b311536d8bb7cfd8bf4eb053649da020d0c9a20dc621071284d9e207c2394"):
            return finish("n128 retained outputs or temporal qualification changed")
        record["n128_input_sha256"] = {"h050": digest(old128), "h025": digest(fine128)}
        baseline = read_reference(old128, 128, .0000005)
        ensure_inputs()
        parity_path = run_reference(128, "0.0000005", "n128-h050")
        record["runs"].append(str(parity_path))
        ensure_inputs()
        comparison = compare(read_reference(parity_path, 128, .0000005), baseline)
        record["n128_same_grid_parity"] = comparison
        metrics = ("binned_force", "sampled_peak_force", "total_impulse", "cumulative_impulse", "sampled_force_trace")
        if any(comparison[name] > 1e-8 for name in metrics):
            return finish("n128 same-grid parity failed")
        coarse_path = run_reference(256, "0.0000005", "n256-h050")
        record["runs"].append(str(coarse_path))
        ensure_inputs()
        coarse = read_reference(coarse_path, 256, .0000005)
        fine_path = run_reference(256, "0.00000025", "n256-h025")
        record["runs"].append(str(fine_path))
        ensure_inputs()
        fine = read_reference(fine_path, 256, .00000025)
        temporal = compare(coarse, fine)
        record["n256_temporal"] = temporal
        if not temporal["passes_refinement"]:
            return finish("n256 temporal refinement failed")
        if digest(fine128) != record["n128_input_sha256"]["h025"]:
            return finish("n128 fine reference changed during continuation")
        reference128 = read_reference(fine128, 128, .00000025)
        spatial = compare(reference128, fine)
        record["n128_to_n256"] = spatial
        record["spatial_qualified"] = spatial["passes_refinement"]
        return finish("n256 spatial comparison completed" if spatial["passes_refinement"] else "n256 spatial refinement failed", 0 if spatial["passes_refinement"] else 1)
    except Exception as exc:
        record["error"] = repr(exc)
        return finish("reference continuation stopped")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if not receipt.exists():
            receipt.write_text(json.dumps({"reason": "unexpected supervisor error", "error": repr(exc)}, indent=2) + "\n")
        raise
