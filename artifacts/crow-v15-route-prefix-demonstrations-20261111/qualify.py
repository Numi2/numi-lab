from pathlib import Path
import gzip,json,hashlib,numpy as np
from metalrobo.mlx_policy_learning import read_policy_rollout_pack
A=Path("artifacts/crow-v15-route-prefix-demonstrations-20261111");P=A/"teacher-20261126";R=Path("artifacts/crow-v15-teacher-perch-balanced-approach-control-20261111/teacher-20261126")
def load(p):
 d={}
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);key=(x["lane"],x["step"])
    if key in d:raise ValueError("duplicate sample "+str(key))
    d[key]=x
 return d
reference,current=load(R),load(P)
ref=json.loads((R/"run.stdout").read_text())["first_episode_navigation"];cur=json.loads((P/"run.stdout").read_text())["first_episode_navigation"]
r=read_policy_rollout_pack(Path("build/crow-v15-route-prefix-demonstrations-20261111/teacher-20261126-rollout.bin"),library_path="build/crow-v15-hip-roll-journey-integration-20261111/libmetalrobo.dylib")
assert (r.control_step_count,r.environment_count,r.action_count)==(3800,8,18)
stages=r.outcome("navigation_waypoints_reached").reshape(3800,8)
labels=r.teacher_actions.reshape(3800,8,18)
rows=[]
for lane in range(8):
 end=min(ref["steps_by_environment"][lane],cur["steps_by_environment"][lane])
 missing=[t for t in range(1,end+1) if (lane,t) not in reference or (lane,t) not in current]
 diff=[t for t in range(1,end+1) if (lane,t) in reference and (lane,t) in current and reference[lane,t]!=current[lane,t]]
 completion=cur["first_completion_step_by_environment"][lane]
 eligible=(not cur["physical_failure_by_environment"][lane] and cur["maximum_waypoints_by_environment"][lane]>=5 and completion is not None and not missing and not diff and ref["steps_by_environment"][lane]==cur["steps_by_environment"][lane])
 endlabel=min(int(completion or 0),end,3800)
 stage_counts={str(k):int(np.sum(stages[400:endlabel,lane]==k)) for k in range(5)} if eligible else {}
 if eligible:
  assert np.isfinite(labels[400:endlabel,lane]).all()
 rows.append({"lane":lane,"physical_failure":cur["physical_failure_by_environment"][lane],"first_episode_steps":cur["steps_by_environment"][lane],"waypoints":cur["maximum_waypoints_by_environment"][lane],"first_fifth_waypoint_step":completion,"missing_endpoint_samples":len(missing),"different_endpoint_samples":len(diff),"first_difference":diff[:1],"eligible_route_prefix":eligible,"stage_counts":stage_counts})
o={"first_episode_outcomes_equal":ref==cur,"lanes":rows,"eligible_lanes":[x["lane"] for x in rows if x["eligible_route_prefix"]],"validation_lanes":[0,6],"scope":"Full-start physically reached route prefixes on one previously used training course and seed. Entire physically failed episodes excluded. No landing labels are qualified. Endpoint comparison does not establish low-level actuator parity; no student-state corrective teacher claim."}
(A/"qualification.json").write_text(json.dumps(o,indent=2)+"\n")
print(json.dumps(o,indent=2))
assert all(l in o["eligible_lanes"] for l in o["validation_lanes"])
assert all(x["eligible_route_prefix"] for x in rows if not x["physical_failure"])
