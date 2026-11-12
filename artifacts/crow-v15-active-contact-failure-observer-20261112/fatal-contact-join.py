from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-active-contact-failure-observer-20261112");D=A/"candidate-20261126-rebound-hip"
meta=json.loads((D/"run.stdout").read_text())["first_episode_navigation"]
with gzip.open(D/"replay.json.gz","rt") as f:names=json.load(f)["payload"]["body_names"]
ends=meta["steps_by_environment"]; reasons=meta["termination_reason_by_environment"]; events=[[] for _ in ends]; states={}
with gzip.open(D/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_ACTIVE_CONTACT "):
   z=line.split();t,k,e,a,b=map(int,z[1:6]);imp=float(z[6])
   if t<=ends[e]+1 and ((a<35 and b>=36) or (b<35 and a>=36)):
    events[e].append(dict(trace_submission=t,substep=k,collider_ids=[a,b],collider_names=[names[a],names[b]],normal_impulse_ns=imp,point_m=list(map(float,z[7:10]))))
  elif line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);states[x["lane"],x["step"]]=x
rows=[]
for e,end in enumerate(ends):
 fatal=[x for x in events[e] if x["trace_submission"]==end+1 and x["substep"]==15] if reasons[e]==3 else []
 context=[]
 for t in range(max(1,end-2),end+2):
  x=states.get((e,t))
  if not x:continue
  q=x["q"];a,b,c,d=q[3:7]
  context.append(dict(trace_submission=t,position_m=q[:3],attitude_roll_pitch_yaw_rad=[math.atan2(2*(d*a+b*c),1-2*(a*a+b*b)),math.asin(max(-1,min(1,2*(d*b-c*a)))),math.atan2(2*(d*c+a*b),1-2*(b*b+c*c))],linear_angular_velocity=x["v"][:6],policy_action=x.get("policy_action")))
 rows.append(dict(lane=e,reported_episode_transitions=end,reason=reasons[e],trace_submission_at_reported_end=end+1 if reasons[e] else None,first_obstacle_impulse=events[e][0] if events[e] else None,final_substep_obstacle_impulses=fatal,preceding_recorded_states=context))
(A/"fatal-contact-join.json").write_text(json.dumps(rows,indent=2)+"\n")
for r in rows:print(r["lane"],"end",r["reported_episode_transitions"],"reason",r["reason"],"first",r["first_obstacle_impulse"],"final_substep",r["final_substep_obstacle_impulses"])
