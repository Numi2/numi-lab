#!/usr/bin/env python3
"""Qualify the near-contact FP64 temporal reference and midpoint trajectories."""
import bisect,json,math,pathlib,sys
p=pathlib.Path(__file__).resolve().parents[2]/"docs/evidence/midpoint-first-failure-20261109"
def read(name):return json.loads((p/name).read_text())
a=read("slider-near-rk4-h050.json");b=read("slider-near-rk4-h025.json")
assert a["initial_gap_m"]==b["initial_gap_m"]==.000095 and a["onset_s"]==b["onset_s"]==0
assert len(a["trace"])==len(b["trace"]) and all(abs(x[0]-y[0])<1e-12 for x,y in zip(a["trace"],b["trace"]))
rel=lambda x,y:abs(x-y)/max(abs(y),1e-30)
limits={"binned_force":.0025,"sampled_peak_force":.01,"total_impulse":.0025,"cumulative_impulse":.01}
temporal={"binned_force":max(abs(x-y) for x,y in zip(a["force_bins_N"],b["force_bins_N"]))/max(b["force_bins_N"]),
 "sampled_peak_force":rel(a["peak_force_N"],b["peak_force_N"]),"total_impulse":rel(a["impulse_Ns"],b["impulse_Ns"]),
 "cumulative_impulse":max(abs(x[2]-y[2]) for x,y in zip(a["trace"],b["trace"]))/b["impulse_Ns"]}
temporal["passes"]=all(temporal[k]<=v for k,v in limits.items()) and min(a["min_gap_m"],b["min_gap_m"])>1e-7
results=[]
for tag in ("015625","0078125"):
 x=read("slider-near-midpoint-h"+tag+".json");assert x["initial_gap_m"]==b["initial_gap_m"] and x["contact_cross_checked"]
 trace=x["cumulative_impulse_trace"];times=[r[0] for r in trace]
 def sample(t):
  i=bisect.bisect_left(times,t)
  if i==0:return trace[0][1]
  if i==len(trace):return trace[-1][1]
  l,r=trace[i-1],trace[i];return l[1]+(r[1]-l[1])*(t-l[0])/(r[0]-l[0])
 row={"h_s":x["h"],"binned_force":max(abs(u-v) for u,v in zip(x["force_bins_N"],b["force_bins_N"]))/max(b["force_bins_N"]),
  "sampled_peak_force":rel(x["sampled_peak_force_N"],b["peak_force_N"]),"total_impulse":rel(x["impulse_Ns"],b["impulse_Ns"]),
  "cumulative_impulse":max(abs(sample(y[0])-y[2]) for y in b["trace"])/b["impulse_Ns"],
  "fem_work_relative_error":rel(x["fem_work_J"],b["fem_contact_work_J"]),"slider_work_relative_error":rel(x["slider_work_J"],b["slider_contact_work_J"]),
  "pair_work_absolute_error_J":abs(x["pair_contact_work_J"]-b["relative_contact_work_J"]),
  "fem_torque_x_relative_error":rel(x["fem_torque_impulse_x_Nms"],b["torque_impulse_x_Nms"]),
  "fem_torque_y_relative_error":rel(x["fem_torque_impulse_y_Nms"],b["torque_impulse_y_Nms"]),
  "max_fem_momentum_residual_Ns":x["max_fem_momentum_residual_Ns"],"max_slider_momentum_residual_Ns":x["max_slider_momentum_residual_Ns"],
  "initial_contact_potential_J":x["initial_contact_potential_J"],"pair_contact_work_J":x["pair_contact_work_J"],
  "work_plus_contact_potential_change_J":x["contact_work_plus_potential_change_J"],"minimum_gap_m":x["min_gap_m"]}
 row["passes"]=all(row[k]<=v for k,v in limits.items()) and row["minimum_gap_m"]>1e-7
 results.append(row)
out={"schema":"numi.slider-near-contact-fp64.v1","scope":"independent CPU only; no native transaction or replay", "initial_gap_m":b["initial_gap_m"],"initial_reference_force_N":b["trace"][0][1],"temporal_reference":temporal,"midpoint_vs_refined_reference":results,"cumulative_interpolation":"linear in midpoint step integrals on the RK4 2 us output clock","limits":limits,"pass_scope":"force/impulse/clearance only; work and torque reported without a new gate"}
print(json.dumps(out,indent=2,sort_keys=True))
sys.exit(0 if temporal["passes"] and all(x["passes"] for x in results) else 1)
