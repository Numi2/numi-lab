from pathlib import Path
import collections,gzip,json,math
A=Path("artifacts/crow-v15-contact-friction-screen-20261111");D=A/"high";s={};by=collections.defaultdict(lambda:[0.0]*37)
with gzip.open(D/"trace.log.gz","rt") as f:
 for line in f:
  z=line.split()
  if z and z[0]=="step":
   e,t=int(z[1]),int(z[2])+1
   if t>450:
    qi=z.index("q");vi=z.index("v");s[e,t]=(list(map(float,z[qi+1:vi])),list(map(float,z[vi+1:])))
with gzip.open(D/"substeps.log.gz","rt") as f:
 for line in f:
  z=line.split()
  if z and z[0]=="CROW_TWO_TOE_SUBSTEP":
   t,k,e=map(int,z[1:4]);
   if t>450:
    for i,v in enumerate(map(float,z[6:])):by[e,t][i]+=v
out=[]
for e in range(8):
 counts=collections.Counter();maximum=0.;minup=1e9
 for t in range(451,551):
  q,vel=s[e,t];v=by[e,t];tilt=math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(x*x for x in vel[:3]));maximum=max(maximum,speed);minup=min(minup,v[0]+v[1])
  checks={"left_and_right_load":min(v[0],v[1])>.0002,"half_weight":v[0]+v[1]>=.5*.45006*9.81*.02,"left_opposition":sum(v[21+i] for i in range(6))>=.0002 and sum(v[21+i] for i in (6,7))<=-.0002,"right_opposition":sum(v[29+i] for i in range(6))>=.0002 and sum(v[29+i] for i in (6,7))<=-.0002,"no_nonfoot":abs(v[4])<=1e-7,"upright":tilt<=.35,"slow":speed<=.12}
  for k,ok in checks.items():counts[k]+=not ok
 out.append({"lane":e,"failed_intervals_by_condition":dict(counts),"max_speed_m_s":maximum,"minimum_foot_upward_impulse_ns":minup})
(A/"high/terminal-diagnostics.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out[2:4],indent=2))
