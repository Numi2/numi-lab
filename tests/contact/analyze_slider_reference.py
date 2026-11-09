#!/usr/bin/env python3
"""Compare independently integrated slider traces and physical invariants."""
import json, pathlib, sys
p=pathlib.Path(__file__).resolve().parents[2]/"docs/evidence/midpoint-first-failure-20261109"
a=json.loads((p/"slider-rk4-h050.json").read_text());b=json.loads((p/"slider-rk4-h025.json").read_text())
if len(a["trace"])!=len(b["trace"]):raise SystemExit("different sample clocks")
force_bin=max(abs(x-y) for x,y in zip(a["force_bins_N"],b["force_bins_N"]))/max(b["force_bins_N"])
peak=abs(a["peak_force_N"]-b["peak_force_N"])/b["peak_force_N"]
impulse=abs(a["impulse_Ns"]-b["impulse_Ns"])/b["impulse_Ns"]
cumulative=max(abs(x[2]-y[2]) for x,y in zip(a["trace"],b["trace"]))/b["impulse_Ns"]
pos=abs(a["slider_final_z_m"]-b["slider_final_z_m"])/max(abs(b["slider_final_z_m"]),1e-8)
vel=abs(a["slider_final_v_mps"]-b["slider_final_v_mps"])/max(abs(b["slider_final_v_mps"]),1e-6)
energy=max(abs(x["final_energy_J"]-x["initial_energy_J"])/abs(x["initial_energy_J"]) for x in (a,b))
momentum=max(abs(x["momentum_balance_error_Ns"]) for x in (a,b))
result={"force_bin_error":force_bin,"sampled_peak_error":peak,"total_impulse_error":impulse,
 "maximum_cumulative_impulse_error":cumulative,"final_slider_position_error":pos,
 "final_slider_velocity_error":vel,"max_energy_relative_drift":energy,
 "max_momentum_balance_error_Ns":momentum,"minimum_gap_m":min(a["min_gap_m"],b["min_gap_m"]),
 "absolute_normalization_floors":{"position_m":1e-8,"velocity_mps":1e-6},
 "passes":force_bin<=.0025 and peak<=.01 and impulse<=.0025 and cumulative<=.01 and pos<=.0025 and vel<=.0025 and energy<=1e-9 and momentum<=1e-10 and min(a["min_gap_m"],b["min_gap_m"])>1e-7 and abs(b["area_m2"]-.0004)<=8e-11}
print(json.dumps(result,indent=2,sort_keys=True));sys.exit(0 if result["passes"] else 1)
