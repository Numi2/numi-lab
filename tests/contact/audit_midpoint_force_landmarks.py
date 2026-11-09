#!/usr/bin/env python3
"""Locate retained native force and cumulative-impulse landmarks by mesh."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "docs/evidence/midpoint-first-failure-20261109"
R = ROOT / "docs/evidence/midpoint-spatial-cost-dynamic-20261108"
OUT = E / "midpoint-force-landmarks-001.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def analyze(n, name, integrity_name):
    paths = {"trace": E / (name + ".json"), "analysis": E / (name + "-analysis.json"),
             "execution": E / (name + "-execution.json"), "reference": R / ("rk4-n%d-h025.json" % n)}
    integrity = json.loads((E / integrity_name).read_text())
    if integrity.get("valid") is not True:
        raise ValueError("retained trace did not pass its separate integrity audit")
    for path in paths.values():
        if integrity["input_sha256"].get(str(path.relative_to(ROOT))) != digest(path):
            raise ValueError("input changed since separate integrity audit: " + str(path))
    d, a, r = (json.loads(paths[k].read_text()) for k in ("trace", "analysis", "reference"))
    h = d["h"]
    rows = d["traces"]
    if (not a["passes"] or len(rows) != round(.01 / h) or d["accepted_steps"] != len(rows)
        or d["rejected_steps"] != 0 or r["mesh_divisions"] != n):
        raise ValueError("incomplete or mismatched retained inputs")
    sums = [0.0]
    bins = [0.0] * 10
    for row in rows:
        j = row["step"]
        sums.append(sums[-1] + row["accepted_normal_impulse"])
        bins[min(9, int((j + .5) * h / .001))] += row["force"] * h / .001
    def impulse_at(t):
        k = min(int(t / h), len(rows) - 1)
        fraction = max(0., min(1., (t - k * h) / h))
        return sums[k] + fraction * (sums[k+1] - sums[k])
    differences = [(abs(impulse_at(t)-j), t, impulse_at(t), j) for t, _, j in r["trace"]]
    delta, time, native_j, reference_j = max(differences)
    peak_i, peak = max(enumerate(rows), key=lambda item: item[1]["force"])
    trace_peak = max(r["trace"], key=lambda item: item[1])
    force_errors = [(abs(x-y), i, x, y) for i, (x, y) in enumerate(zip(bins, r["force_bins_N"]))]
    bdelta, bi, bnative, bref = max(force_errors)
    if (not math.isclose(delta / r["impulse_Ns"], a["cumulative_impulse_error"], rel_tol=1e-11) or
        not math.isclose(bdelta / max(r["force_bins_N"]), a["binned_force_error"], rel_tol=1e-11) or
        not math.isclose(peak["force"], a["native_peak_force_N"], rel_tol=1e-14)):
        raise ValueError("landmarks do not reproduce the retained analysis")
    return {"mesh_divisions": n, "nodes": d["nodes"], "timestep_s": h,
        "native_peak_force_N": peak["force"], "native_peak_stage_time_s": (peak_i+.5)*h,
        "reference_integration_step_sampled_peak_N": r["peak_force_N"],
        "reference_peak_time": "not retained at the integration-step sampling rate",
        "reference_2us_trace_sample_maximum_N": trace_peak[1],
        "reference_2us_trace_sample_maximum_time_s": trace_peak[0],
        "peak_relative_error": a["peak_force_error"],
        "worst_1ms_bin": bi, "worst_bin_start_s": bi*.001,
        "worst_bin_native_N": bnative, "worst_bin_reference_N": bref,
        "worst_bin_error_normalized_by_reference_peak_bin": a["binned_force_error"],
        "max_cumulative_error_reference_time_s": time,
        "cumulative_native_Ns": native_j, "cumulative_reference_Ns": reference_j,
        "cumulative_max_error_Ns": delta, "cumulative_error_normalized_by_total_reference_impulse": a["cumulative_impulse_error"],
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in paths.values()},
        "integrity_audit": integrity_name, "integrity_audit_sha256": digest(E / integrity_name)}


output = {"schema": "numi.midpoint-force-landmarks.v1", "fixtures": [
    analyze(2, "stage-ladder-n2-h015625", "midpoint-strict-integrity-001.json"),
    analyze(8, "stage-ladder-n8-h0078125", "midpoint-n8-strict-integrity-001.json")],
    "scope": "Retained complete native fixed-plane steps versus their own mesh-specific CPU references. Neither spatial convergence nor a performance result is established."}
OUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
print(json.dumps(output, indent=2))
