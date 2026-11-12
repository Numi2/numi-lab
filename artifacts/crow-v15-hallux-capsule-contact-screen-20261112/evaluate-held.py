from pathlib import Path
import json,gzip
p=Path("artifacts/crow-v15-hallux-capsule-contact-screen-20261112");c=json.loads((p/"candidate-vs-held-control-20261127-comparison.json").read_text());s=json.loads((p/"analysis-held-20261127.json").read_text());rows=c["rows"];ratios=[x["candidate_travel_m"]/x["reference_travel_m"] for x in rows];drag=s["sole_clear_toe_low_counts"]
checks={"comparison":c["all_gates_pass"],"minimum_three_clearance_bouts":all(min(x["candidate_swing_counts"])>=3 for x in rows),"no_more_unloaded_per_lane":all(x["candidate_unloaded_substeps"]<=x["reference_unloaded_substeps"] for x in rows),"zero_toe_caps":all(x["candidate_caps"]==0 for x in rows),"all_lanes_90_percent_travel":all(x>=.9 for x in ratios),"toe_low_at_most_120_percent":drag["sham"]>0 and drag["candidate"]<=1.2*drag["sham"],"contact_point_speed":s["contact_speed_gate"]}
def prefix200(path):
 states={};sub={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1])
    if x["step"]==200:states[x["lane"]]=[x[k] for k in ("q","v","policy_action","contacts")]
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);
    if t==200:sub[e,k]=z[4:]
 assert len(states)==8 and len(sub)==128
 return states,sub
paths=[Path("artifacts/crow-v15-small-support-stroke-20261112/sham-20261127-paired"),p/"candidate-20261127-rebound"]
step200_equal=prefix200(paths[0])==prefix200(paths[1])
checks["complete_first_episode"]=all((lambda e:e["steps_by_environment"]==[300]*8 and not any(e["closed_by_environment"]) and not any(e["physical_failure_by_environment"]))(json.loads((path/"run.stdout").read_text())["first_episode_navigation"]) for path in paths)
out={"checks":checks,"passed":all(checks.values()),"step_200_state_equal_diagnostic_only":step200_equal,"ratios":ratios,"toe_low":drag,"contact_speed":{k:v["all"] for k,v in s["arms"].items()},"limits":"Previously used development seed; two seconds walking. Endpoint toe clearance and contact point velocities are sampled measures, not biological qualification."};(p/"evaluation-held.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
