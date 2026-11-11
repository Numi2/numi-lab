from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-flight-moment-ablation-screen-20261112");B=Path("artifacts/crow-v15-toe-body-extended-turn-screen-20261111")
def read(p):
 out={}
 with gzip.open(p/"flight.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_FLIGHT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);key=x["lane"],x["step"]
    if key in out:raise ValueError("duplicate flight sample")
    out[key]=x
 return out
cur=read(A);ref=read(B);prefix=[k for k in sorted(set(ref)|set(cur)) if k[1]<450];mismatch=[k for k in prefix if ref.get(k)!=cur.get(k)]
a=json.loads((A/"analysis.json").read_text());b=json.loads((B/"analysis.json").read_text());rows=[]
for x,y in zip(a["rows"],b["rows"]):
 assert x["lane"]==y["lane"];e=x["lane"];common=sorted(set(k[1] for k in cur if k[0]==e and 450<=k[1]<=1600)&set(k[1] for k in ref if k[0]==e));diff=[]
 for t in common:
  p=cur[e,t]["q"];q=ref[e,t]["q"];diff.append({"step":t,"position_difference_m":math.dist(p[:3],q[:3])})
 rows.append({"lane":e,"ablated_command":"yaw" if e<6 else "pitch","target_heading_rad":x["target_heading_rad"],"candidate_end":x["first_episode_steps"],"physical_failure":x["physical_failure"],"candidate_ground_substeps_after400":x["ground_contact_substeps_from_step400"],"candidate_toe_cap_substeps_after400":x["toe_caps_from_step400"],"missing_contact_substeps":x["missing_or_duplicate_substeps"],"candidate_last_sample_heading_error":x["final_sample_heading_error_rad"],"reference_heading_error_at_same_last_sample":next((abs(math.atan2(math.sin(y["target_heading_rad"]-z["yaw_rad"]),math.cos(y["target_heading_rad"]-z["yaw_rad"]))) for z in y["samples"] if x["samples"] and z["step"]==x["samples"][-1]["step"]),None),"common_sample_position_differences":diff})
out={"prefix_sample_count":len(prefix),"prefix_mismatches":mismatch,"rows":rows,"candidate_errors":a["substep_errors"],"overflow_count":len(a["contact_overflows"]),"failed_environment_steps":a["failed_environment_steps"],"limits":"development-only matched starts; command ablation retains response filter and other body moment, actual wrenches unmeasured; sparse kinematic samples are not continuous or aerodynamic/biological qualification"};(A/"comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
