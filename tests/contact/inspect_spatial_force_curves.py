#!/usr/bin/env python3
"""Inspect aligned retained n32/n64 force and impulse traces without fitting."""
import hashlib,json,math,pathlib
p=pathlib.Path(__file__).resolve().parents[2]/"docs/evidence"
paths=[p/"midpoint-spatial-cost-dynamic-20261108/rk4-n32-h025.json",p/"midpoint-first-failure-20261109/rk4-n64-parallel-h025.json"]
a,b=[json.loads(x.read_text()) for x in paths]
assert [a["mesh_divisions"],b["mesh_divisions"]]==[32,64]
assert a["h"]==b["h"]==.00000025 and len(a["trace"])==len(b["trace"])==5001
assert all(abs(x[0]-y[0])<1e-12 and all(math.isfinite(z) for z in x+y) for x,y in zip(a["trace"],b["trace"]))
peak=lambda x:max(x["trace"],key=lambda row:row[1])
worst_force=max(zip(a["trace"],b["trace"]),key=lambda rows:abs(rows[0][1]-rows[1][1]))
worst_impulse=max(zip(a["trace"],b["trace"]),key=lambda rows:abs(rows[0][2]-rows[1][2]))
record=lambda rows:{"time_s":rows[0][0],"n32":rows[0][1:],"n64":rows[1][1:]}
out={"schema":"numi.spatial-force-curves.v1","scope":"exploratory inspection of retained CPU reference; not native qualification or an identified mechanism",
 "input_sha256":{str(q.relative_to(p)):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths},
 "n32_peak_over_integrator_samples_N":a["peak_force_N"],"n64_peak_over_integrator_samples_N":b["peak_force_N"],
 "n32_peak_at_2us_output_clock":peak(a),"n64_peak_at_2us_output_clock":peak(b),
 "largest_aligned_force_difference":record(worst_force),"largest_aligned_cumulative_impulse_difference":record(worst_impulse),
 "bins":[{"start_s":i*.001,"n32_N":x,"n64_N":y,"n32_minus_n64_N":x-y} for i,(x,y) in enumerate(zip(a["force_bins_N"],b["force_bins_N"]))],
 "normalization":"peak and cumulative errors normalized by n64 full-step peak and final total impulse in the retained spatial gate; 2 us aligned data shown here"}
print(json.dumps(out,indent=2))
