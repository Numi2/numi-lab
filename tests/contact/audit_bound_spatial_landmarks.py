#!/usr/bin/env python3
"""Inspect hash-bound n64/n128 force peaks and cumulative impulses."""
import hashlib
import json
from pathlib import Path

from large_spatial_reference import read_reference, compare

root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
a_path = e / "rk4-n64-rebound-006-h025.json"
b_path = e / "rk4-n128-retry2-h025.json"
analysis_path = e / "spatial-n128-retry2-analysis.json"
bound_path = e / "spatial-n64-rebound-006-analysis.json"
execution_path = e / "rk4-n64-rebound-006-h025-execution.json"
supervisor_path = e / "n64-rebound-006-supervisor.json"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

reference = json.loads(analysis_path.read_text())
bound = json.loads(bound_path.read_text())
execution = json.loads(execution_path.read_text())
supervisor = json.loads(supervisor_path.read_text())
if (execution.get("returncode") != 0 or execution.get("stopped_reason") is not None or
        execution.get("output_sha256") != digest(a_path) or
        supervisor.get("analysis_returncode") != 0 or
        supervisor.get("analysis_sha256") != digest(bound_path) or
        not bound.get("spatial_test_valid") or
        not bound["n64_temporal"]["passes"] or
        not bound["n128_temporal"]["passes"]):
    raise ValueError("hash-bound n64 repeat or temporal qualification invalid")
if not reference["spatial_test_valid"] or reference["inputs_sha256"]["h025"] != digest(b_path):
    raise ValueError("n128 reference or temporal qualification changed")
a = read_reference(a_path, 64, 2.5e-7)
b = read_reference(b_path, 128, 2.5e-7)
comparison = compare(a, b)
expected = reference["n64_to_128"]
for new, old in (("binned_force", "binned_force"), ("sampled_peak_force", "peak_force"),
                 ("total_impulse", "total_impulse"), ("cumulative_impulse", "cumulative_impulse")):
    if abs(comparison[new] - expected[old]) > 1e-14:
        raise ValueError("refinement analysis mismatch: " + new)
peak64 = max(a["trace"], key=lambda row: row[1])
peak128 = max(b["trace"], key=lambda row: row[1])
maximum_impulse = max(zip(a["trace"], b["trace"]), key=lambda pair: abs(pair[0][2] - pair[1][2]))
index = max(range(10), key=lambda i: abs(a["force_bins_N"][i] - b["force_bins_N"][i]))
result = {
    "scope": "independent CPU fixture, hash-bound n64 repeat; not a native Matter rollout or material calibration",
    "inputs_sha256": {str(p): digest(p) for p in (a_path, b_path, analysis_path, bound_path, execution_path, supervisor_path)},
    "step_sampled_peak_force_N": {"n64": a["peak_force_N"], "n128": b["peak_force_N"],
                                  "sample_interval_seconds": 2.5e-7},
    "two_microsecond_trace_peaks": {"n64_time_s": peak64[0], "n64_force_N": peak64[1],
                                     "n128_time_s": peak128[0], "n128_force_N": peak128[1],
                                     "note": "times and forces are sampled on the 2 µs trace; they need not be the stepwise peaks"},
    "largest_1ms_bin_difference": {"start_s": index * .001, "end_s": (index + 1) * .001,
                                    "n64_force_N": a["force_bins_N"][index], "n128_force_N": b["force_bins_N"][index]},
    "largest_cumulative_impulse_difference": {"time_s": maximum_impulse[0][0],
                                               "n64_Ns": maximum_impulse[0][2], "n128_Ns": maximum_impulse[1][2]},
    "comparison": comparison,
}
print(json.dumps(result, indent=2, sort_keys=True))
