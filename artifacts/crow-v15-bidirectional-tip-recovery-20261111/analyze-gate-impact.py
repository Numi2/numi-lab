from pathlib import Path
import gzip,json,math
D=Path("artifacts/crow-v15-bidirectional-tip-recovery-20261111/autonomous-20261126");d=json.loads((D/"run.stdout").read_text());end=d["first_episode_navigation"]["steps_by_environment"][5]
with gzip.open(D/"replay.json.gz","rt") as f:names=json.load(f)["payload"]["body_names"]
states={};hits=[]
with gzip.open(D/"run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(l.split(" ",1)[1]);t=x["step"]
   if x["lane"]==5 and 3700<=t<=end+1:states[t]=x
  elif l.startswith("CROW_SCENE_CONTACT "):
   z=l.split();t,k,e,a,b=map(int,z[1:6]);n=float(z[6])
   if e==5 and 3700<=t<=end+1 and n>1e-7 and 36 in (a,b):hits.append({"step":t,"substep":k,"body_a":names[a],"body_b":names[b],"normal_impulse_Ns":n})
assert hits
first=min(hits,key=lambda x:(x["step"],x["substep"]));prev=states[first["step"]-1];q=prev["q"];a,b,c,w=q[3:7];att=[math.atan2(2*(w*a+b*c),1-2*(a*a+b*b)),math.asin(max(-1,min(1,2*(w*b-c*a)))),math.atan2(2*(w*c+a*b),1-2*(b*b+c*c))]
out={"first_episode_end":end,"termination_reason":d["first_episode_navigation"]["termination_reason_by_environment"][5],"first_recorded_gate_contact":first,"preceding_endpoint":{"step":prev["step"],"position_m":q[:3],"roll_pitch_yaw_rad":att,"linear_and_angular_velocity":prev["v"][:6],"policy_action":prev["policy_action"]},"limits":"Logged policy outputs are not the applied actuator targets; those were not recorded. The preceding sample is a control endpoint, not the exact impact instant. Contact detection is limited to the instrumented substep data and this used development course."};(D/"gate-impact.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
