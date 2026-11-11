from pathlib import Path
import gzip,json,math
D=Path("artifacts/crow-v15-bidirectional-tip-recovery-20261111/autonomous-20261126");d=json.loads((D/"run.stdout").read_text())["first_episode_navigation"];ends=d["steps_by_environment"];reason=d["termination_reason_by_environment"]
with gzip.open(D/"replay.json.gz","rt") as f:names=json.load(f)["payload"]["body_names"]
states={};hits={2:[],5:[]}
with gzip.open(D/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
   if e in hits and t==ends[e]:states[e]=x
  elif line.startswith("CROW_SCENE_CONTACT "):
   z=line.split();t,k,e,a,b=map(int,z[1:6]);normal=float(z[6])
   if e in hits and t==ends[e]+1 and normal>1e-7:hits[e].append({"step":t,"substep":k,"body_a":names[a],"body_b":names[b],"normal_impulse_Ns":normal})
rows=[]
for e in (2,5):
 assert reason[e]==3 and e in states and hits[e]
 x=states[e];q=x["q"];a,b,c,w=q[3:7];rpy=[math.atan2(2*(w*a+b*c),1-2*(a*a+b*b)),math.asin(max(-1,min(1,2*(w*b-c*a)))),math.atan2(2*(w*c+a*b),1-2*(b*b+c*c))];first=min(hits[e],key=lambda z:z["substep"])
 rows.append({"lane":e,"first_episode_end":ends[e],"termination_reason":reason[e],"first_recorded_scene_contact_in_terminal_interval":first,"preceding_endpoint":{"step":x["step"],"position_m":q[:3],"roll_pitch_yaw_rad":rpy,"linear_and_angular_velocity":x["v"][:6],"policy_action":x["policy_action"]}})
out={"rows":rows,"limitations":"Contact threshold and first sampled substep do not establish continuous exact time. Endpoint precedes logged terminal interval by one; policy output is not the applied actuator target. This is a development course and does not establish biological realism."};(D/"contact-failures.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
