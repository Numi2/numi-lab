#!/usr/bin/env python3
"""Original temporal gates for the near-band initial state."""
import json,math,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[2]
p=root/"docs/evidence/midpoint-first-failure-20261109"
a,b=(json.loads((p/f"near-rk4-n8-h{t}.json").read_text()) for t in ("050","025"))
for x in (a,b):
 assert x["mesh_divisions"]==8 and x["duration"]==.01 and x["initial_gap_m"]==.000095
 assert len(x["trace"])==5001 and abs(x["area_m2"]-.0004)<1e-14
 assert x["min_gap_m"]>1e-7 and x["impulse_Ns"]>0 and x["peak_force_N"]>0
assert all(abs(u[0]-v[0])<1e-12 for u,v in zip(a["trace"],b["trace"]))
r={"binned_force":max(abs(u-v) for u,v in zip(a["force_bins_N"],b["force_bins_N"]))/max(b["force_bins_N"]),
"total_impulse":abs(a["impulse_Ns"]-b["impulse_Ns"])/b["impulse_Ns"],
"peak_force":abs(a["peak_force_N"]-b["peak_force_N"])/b["peak_force_N"],
"cumulative_impulse":max(abs(u[2]-v[2]) for u,v in zip(a["trace"],b["trace"]))/b["impulse_Ns"]}
r["passes"]=r["binned_force"]<=.0025 and r["total_impulse"]<=.0025 and r["peak_force"]<=.01 and r["cumulative_impulse"]<=.01
r.update({"initial_gap_m":b["initial_gap_m"],"initial_force_N":b["trace"][0][1],"reference_peak_includes_initial":True,"minimum_gap_m":b["min_gap_m"]})
print(json.dumps(r,indent=2));sys.exit(0 if r["passes"] else 1)
