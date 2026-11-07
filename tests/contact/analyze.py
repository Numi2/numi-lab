#!/usr/bin/env python3
"""Qualify retained complete-step contact evidence without rewarding early exit."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("evidence",type=Path);a=p.parse_args()
rows=[];runs={}
for entry in json.loads((a.evidence/"attempt-ledger.json").read_text()):
 x=json.loads((a.evidence/(entry["name"]+".json")).read_text());runs[entry["name"]]=x
 finite=all(math.isfinite(t[k]) for t in x["traces"] for k in ["force","normal_impulse","gap","height","contact_work_J"])
 rows.append({"name":entry["name"],"arm":x["arm"],"h":x["h"],"divisions":x["divisions"],
  "accepted_seconds":x["accepted_simulated_seconds"],"rejected_steps":x["rejected_steps"],
  "wall_seconds_including_rejected_attempt":x["wall_seconds"],"simulated_seconds_per_wall_second":x["accepted_simulated_seconds_per_wall_second"],
  "complete_duration":x["rejected_steps"]==0 and abs(x["accepted_simulated_seconds"]-.01)<1e-12,
  "minimum_accepted_gap":x["minimum_gap"],"contact_work_J":x["contact_work_J"],
  "impulse_midvelocity_work_J":x["contact_impulse_midvelocity_work_J"],
  "instrument_qualified":x["law_max_relative_error"]<=1e-4 and x["momentum_max_absolute_error_Ns"]<=1e-7,
  "replay":x["restore_replay"],"rollback":x["forced_rollback_exact"],
  "identity_rejection":x["identity_mismatch_rejected"],"finite":finite,
  "law_max_relative_error":x["law_max_relative_error"],"momentum_max_error_Ns":x["momentum_max_absolute_error_Ns"],
  "maximum_accepted_multiplier_impulse_Ns":x["maximum_accepted_multiplier_impulse_Ns"]})
def bins(x):
 out=[0.]*10
 for t in x["traces"]:
  if t["accepted"]:out[min(9,(round((t["step"]+1)*x["h"]*1e9)-1)//1000000)]+=t["normal_impulse"]/.001
 return out
refs=[runs.get("ref-n2-h125000-ipc"),runs.get("ref-n4-h62500-ipc")]
reference_complete=all(x and x["rejected_steps"]==0 for x in refs)
ref={"complete":reference_complete,"force_max_normalized_change":None,"impulse_relative_change":None,"qualified":False}
if reference_complete:
 av,bv=map(bins,refs)
 ref.update(force_max_normalized_change=max(abs(x-y) for x,y in zip(av,bv))/max(max(bv),1e-6),
  impulse_relative_change=abs(sum(av)-sum(bv))/max(abs(sum(bv)),.000001),
  final_com_change_m=abs(refs[0]["traces"][-1]["com_z"]-refs[1]["traces"][-1]["com_z"]))
 ref["qualified"]=ref["force_max_normalized_change"]<=.0025 and ref["impulse_relative_change"]<=.0025
matched=[]
for h in [250000,500000,1000000,2000000]:
 for n in [1,2]:
  x=[runs[f"pair{k}-n{n}-h{h}-{arm}"] for k,arm in enumerate(["ipc","al","al","ipc"])]
  same=all(t["source_fingerprint"]==x[0]["source_fingerprint"] for t in x)
  different=x[0]["device_fingerprint"]!=x[1]["device_fingerprint"]
  matched.append({"h_ns":h,"divisions":n,"same_source_physics":same,"different_runtime_law_identity":different,
   "same_duration_completed":all(t["rejected_steps"]==0 for t in x),"accuracy_qualified_total_time_ratio":None})
print(json.dumps({"schema":"numi.matter.plane-analysis.v1","runs":rows,"matched":matched,"reference":ref,
 "primary_10_percent_prediction":"inconclusive: reference and complete-duration gates fail",
 "decision":"keep experimental law disabled; force accuracy and matched-duration throughput not qualified",
 "work_definition":"sum final normal force dot accepted nodal displacement; midpoint impulse-velocity work reported separately; rejected attempts contribute no accepted physical work",
 "limits":["An accepted solver status alone does not establish force accuracy.","Contact samples on rejected attempts are trial diagnostics.","A truncated rollout cannot establish a speed improvement.","Both arms retain identical strict final geometric predicates."]},indent=2))
