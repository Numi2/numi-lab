#!/usr/bin/env python3
"""Inspect retained common-clock curves; do not infer a cause of mesh error."""
import json,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
a=json.loads((root/"docs/evidence/midpoint-spatial-cost-dynamic-20261108/rk4-n32-h025.json").read_text())
b=json.loads((root/"docs/evidence/midpoint-first-failure-20261109/rk4-n64-parallel-h025.json").read_text())
assert len(a["trace"])==len(b["trace"])==5001
assert all(x[0]==y[0] for x,y in zip(a["trace"],b["trace"]))
peak_a=max(a["trace"],key=lambda row:row[1]);peak_b=max(b["trace"],key=lambda row:row[1])
diff=max(zip(a["trace"],b["trace"]),key=lambda pair:abs(pair[0][2]-pair[1][2]))
force=max(zip(a["trace"],b["trace"]),key=lambda pair:abs(pair[0][1]-pair[1][1]))
print(json.dumps({"sampling_seconds":2e-6,"n32_peak_trace_time_s":peak_a[0],"n32_peak_trace_force_N":peak_a[1],"n32_peak_every_reference_step_N":a["peak_force_N"],"n64_peak_trace_time_s":peak_b[0],"n64_peak_trace_force_N":peak_b[1],"n64_peak_every_reference_step_N":b["peak_force_N"],"max_cumulative_impulse_difference_time_s":diff[0][0],"n32_cumulative_impulse_Ns":diff[0][2],"n64_cumulative_impulse_Ns":diff[1][2],"max_instantaneous_force_difference_time_s":force[0][0],"n32_force_N":force[0][1],"n64_force_N":force[1][1],"n32_force_bins_N":a["force_bins_N"],"n64_force_bins_N":b["force_bins_N"]},indent=2))
