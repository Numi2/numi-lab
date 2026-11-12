from pathlib import Path
from collections import defaultdict
import gzip,json,math
A=Path("artifacts/crow-v15-uniform-flight-turn-screen-20261112");D=A/"candidate-20261126-rebound-hip"
nav=json.loads((D/"run.stdout").read_text())["first_episode_navigation"]
ends=nav["steps_by_environment"]; by=defaultdict(list)
with gzip.open(D/"run.stderr.gz","rt") as f:
 for line in f:
  if not line.startswith("CROW_SUPPORT_SAMPLE "):continue
  x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
  if not 2<=t<=ends[e]+1:continue
  q=x["q"];v=x["v"];a,b,c,d=q[3:7]
  yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c))
  by[e].append({"episode_transition":t-1,"yaw_rad":yaw,"height_m":q[2],"forward_speed_m_s":v[0]*math.cos(yaw)+v[1]*math.sin(yaw),"position_m":q[:3]})
def window(seq,lo,hi):
 rr=[x for x in seq if lo<=x["episode_transition"]<hi]
 if not rr:return None
 return {"samples":len(rr),**{k:sum(x[k] for x in rr)/len(rr) for k in ("yaw_rad","height_m","forward_speed_m_s")}}
rows=[]
targets=[0,0,.15,-.15,.30,-.30,.45,-.45]
for e in range(8):
 seq=by[e]
 rows.append({"lane":e,"target_rad":targets[e],"steps":ends[e],"reason":nav["termination_reason_by_environment"][e],"physical_failure":nav["physical_failure_by_environment"][e],"sample_count":len(seq),"initial":window(seq,550,600),"turn":window(seq,800,850),"recovery":window(seq,1100,1150),"end":seq[-1] if seq else None})
(A/"analysis.json").write_text(json.dumps(rows,indent=2)+"\n")
for r in rows:print(r["lane"],"target",r["target_rad"],"end",r["steps"],"reason",r["reason"],"coverage",r["sample_count"],"turn",r["turn"],"recovery",r["recovery"])
