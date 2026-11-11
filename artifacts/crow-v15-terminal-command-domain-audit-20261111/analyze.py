from pathlib import Path
import gzip,json,math,statistics
R=Path("artifacts/crow-v15-walk-hold-localized-route-20261111/autonomous-20261126");D=Path("artifacts/crow-v15-terminal-command-domain-audit-20261111");first=json.loads((R/"run.stdout").read_text())["first_episode_navigation"]; ends=first["steps_by_environment"];starts=first["first_completion_step_by_environment"]; phase={};states={}
with gzip.open(R/"run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_PHASE_SAMPLE "):
   a=l.split();t,e=int(a[1]),int(a[2]); p=float(a[4]);
   if starts[e]<=t<=ends[e] and .5<p<1.5:phase[e,t]=float(a[7])
  elif l.startswith("CROW_SUPPORT_SAMPLE "):
   a=json.loads(l.split(" ",1)[1]);e=a["lane"];t=a["step"]
   if starts[e]<=t<=ends[e]:states[e,t]=a
rows=[]
for e in range(8):
 values=[]
 for (lane,t),ex in phase.items():
  if lane!=e or (e,t) not in states:continue
  x=states[e,t];a,b,c,w=x["q"][3:7];yaw=math.atan2(2*(w*c+a*b),1-2*(b*b+c*c)); v=x["v"];vx=math.cos(yaw)*v[0]+math.sin(yaw)*v[1]; speed=max(-.1,min(.35,.55*(ex+.02)+.10-.65*vx));values.append(speed)
 rows.append({"lane":e,"matched_endpoint_samples":len(values),"range_m_s":[min(values),max(values)] if values else [],"count_below_qualified_command_min_0_15":sum(v<.15 for v in values),"count_negative":sum(v<0 for v in values),"median_m_s":statistics.median(values) if values else None})
out={"rows":rows,"limitations":"Reconstructed command from accepted endpoint pose, velocities and logged final-goal error at same numbered samples; timing must be confirmed before treating as applied command. Frozen flight screen covered independent commands at 0.15–0.85 m/s under specified disturbances, not arbitrary combined commands; outside that screen is unqualified, not evidence of instability by itself."};(D/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
