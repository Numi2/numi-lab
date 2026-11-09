#!/usr/bin/env python3
"""Check the compact reference against a retained, frozen n32 full trajectory.

This verifies evaluator agreement, not temporal accuracy or spatial convergence.
The original trace and both input hashes are retained independently.
"""
import hashlib, json, pathlib, shutil, subprocess, time

root = pathlib.Path(__file__).resolve().parents[2]
evidence = root / "docs/evidence/midpoint-first-failure-20261109"
prior = root / "docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n32-h050.json"
exe = pathlib.Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109/evidence-compact-005/numi_contact_fem_fp64_reference_compact")
parity = json.loads((evidence / "compact-reference-trajectory-parity-006.json").read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if sha(exe) != parity["input_sha256"][str(exe)]:
    raise SystemExit("compact executable is not the parity-checked executable")
paths = {s: evidence / ("compact-reference-retained-n32-parity-001" + s) for s in (".json", ".log", "-analysis.json")}
if any(p.exists() for p in paths.values()):
    raise SystemExit("attempt already exists")
if shutil.disk_usage(evidence).free < int(3.2 * 1024**3):
    raise SystemExit("free-space guard")
command = [str(exe), "32", "0.0000005", "0.01"]
start = time.monotonic()
peak_kib = 0
reason = None
with paths[".json"].open("x") as out, paths[".log"].open("x") as err:
    process = subprocess.Popen(command, stdout=out, stderr=err)
    while process.poll() is None:
        sample = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True)
        try:
            peak_kib = max(peak_kib, int(sample.stdout.strip()))
        except ValueError:
            pass
        if peak_kib > 1024 * 1024:
            reason = "one-GiB RSS guard"
        elif shutil.disk_usage(evidence).free < int(3.2 * 1024**3):
            reason = "free-space guard"
        elif time.monotonic() - start > 3600:
            reason = "one-hour elapsed-time guard"
        if reason:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        time.sleep(2)
    code = process.wait()
result = {"scope": "complete n32 trajectory parity at the same h; not a spatial or temporal qualification or timing benchmark",
          "command": command, "compact_binary_sha256": sha(exe), "retained_reference_sha256": sha(prior),
          "compact_source_sha256": sha(root / "tests/contact/fem_plane_fp64_reference_compact.cpp"),
          "included_source_sha256": sha(root / "tests/contact/fem_plane_fp64_reference.cpp"),
          "returncode": code, "stopped_reason": reason, "sampled_peak_rss_kib": peak_kib,
          "wall_seconds_observational_only": time.monotonic() - start}
try:
    a, b = json.loads(prior.read_text()), json.loads(paths[".json"].read_text())
    assert a["mesh_divisions"] == b["mesh_divisions"] == 32 and a["h"] == b["h"] and a["duration"] == b["duration"]
    assert len(a["trace"]) == len(b["trace"]) and all(x[0] == y[0] for x, y in zip(a["trace"], b["trace"]))
    assert len(a["contact_area_weights_m2"]) == len(b["contact_area_weights_m2"])
    f, impulse = a["peak_force_N"], a["impulse_Ns"]
    result.update({"max_force_difference_over_reference_peak": max(abs(x[1] - y[1]) for x, y in zip(a["trace"], b["trace"])) / f,
                   "max_cumulative_impulse_difference_over_reference_total": max(abs(x[2] - y[2]) for x, y in zip(a["trace"], b["trace"])) / impulse,
                   "peak_force_relative_difference": abs(a["peak_force_N"] - b["peak_force_N"]) / f,
                   "total_impulse_relative_difference": abs(a["impulse_Ns"] - b["impulse_Ns"]) / impulse,
                   "max_contact_weight_abs_difference_m2": max(abs(x-y) for x,y in zip(a["contact_area_weights_m2"], b["contact_area_weights_m2"])),
                   "total_area_difference_m2": abs(a["area_m2"] - b["area_m2"]),
                   "total_mass_difference_kg": abs(a["mass_kg"] - b["mass_kg"])})
    result["passes"] = (code == 0 and reason is None and
        all(result[k] <= 1e-8 for k in ("max_force_difference_over_reference_peak", "max_cumulative_impulse_difference_over_reference_total", "peak_force_relative_difference", "total_impulse_relative_difference")) and
        result["max_contact_weight_abs_difference_m2"] <= 1e-18 and result["total_area_difference_m2"] <= 1e-15 and result["total_mass_difference_kg"] <= 1e-12)
except (ValueError, KeyError, AssertionError, ZeroDivisionError) as error:
    result["passes"] = False
    result["analysis_error"] = repr(error)
paths["-analysis.json"].write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
raise SystemExit(0 if result["passes"] else 1)
