#!/usr/bin/env python3
"""Compare FP64 discrete midpoint to refined RK4, not native Matter."""
import bisect, json, pathlib, sys
p=pathlib.Path(__file__).resolve().parents[2]/"docs/evidence/midpoint-first-failure-20261109"
r=json.loads((p/"slider-v2-rk4-h025.json").read_text())
reference_test=json.loads((p/"slider-reference-analysis.json").read_text())
if not reference_test["passes"]: raise SystemExit("independent RK4 reference failed refinement")
results=[]
for tag in ("015625","0078125"):
 a=json.loads((p/("slider-midpoint-fp64-h"+tag+".json")).read_text())
 if not a["contact_cross_checked"]: raise SystemExit("contact tangent check absent")
 trace=a["cumulative_impulse_trace"]; times=[x[0] for x in trace]
 def sample(t):
  j=bisect.bisect_left(times,t)
  if j==0:return trace[0][1]
  if j==len(trace):return trace[-1][1]
  x,y=trace[j-1],trace[j]
  return x[1]+(y[1]-x[1])*(t-x[0])/(y[0]-x[0])
 err=lambda x,y: abs(x-y)/max(abs(y),1e-30)
 row={"h_s":a["h"],
  "binned_force_relative_error":max(abs(x-y) for x,y in zip(a["force_bins_N"],r["force_bins_N"]))/max(r["force_bins_N"]),
  "sampled_peak_force_relative_error":err(a["sampled_peak_force_N"],r["peak_force_N"]),
  "total_impulse_relative_error":err(a["impulse_Ns"],r["impulse_Ns"]),
  "cumulative_impulse_relative_error":max(abs(sample(x[0])-x[2]) for x in r["trace"])/r["impulse_Ns"],
  "slider_position_relative_error":err(a["slider_z_m"],r["slider_final_z_m"]),
  "slider_velocity_relative_error":err(a["slider_v_mps"],r["slider_final_v_mps"]),
  "fem_torque_x_relative_error":err(a["fem_torque_impulse_x_Nms"],r["torque_impulse_x_Nms"]),
  "fem_torque_y_relative_error":err(a["fem_torque_impulse_y_Nms"],r["torque_impulse_y_Nms"]),
  "fem_work_relative_error":err(a["fem_work_J"],r["fem_contact_work_J"]),
  "slider_work_relative_error":err(a["slider_work_J"],r["slider_contact_work_J"]),
  "pair_work_absolute_error_J":abs(a["pair_contact_work_J"]-r["relative_contact_work_J"]),
  "minimum_gap_m":a["min_gap_m"],
  "max_fem_momentum_residual_Ns":a["max_fem_momentum_residual_Ns"],
  "max_slider_momentum_residual_Ns":a["max_slider_momentum_residual_Ns"],
  "max_total_momentum_residual_Ns":a["max_total_momentum_residual_Ns"],
  "contact_cross_symmetry_relative_error":a["contact_cross_symmetry_relative_error"]}
 row["comparison_gates_pass"]=(row["binned_force_relative_error"]<=.0025 and row["sampled_peak_force_relative_error"]<=.01 and row["total_impulse_relative_error"]<=.0025 and row["cumulative_impulse_relative_error"]<=.01 and row["slider_position_relative_error"]<=.0025 and row["slider_velocity_relative_error"]<=.0025 and row["minimum_gap_m"]>1e-7)
 results.append(row)
print(json.dumps({"schema":"numi.slider-fp64-midpoint-comparison.v1","reference":"slider-v2-rk4-h025.json","cumulative_interpolation":"linear interpolation of midpoint step integrals at the 2 us reference clock","scope":"independent CPU discrete comparison only; neither a native Matter step nor a replay test", "results":results},indent=2,sort_keys=True))

sys.exit(0 if all(row["comparison_gates_pass"] for row in results) else 1)
