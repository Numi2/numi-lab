from pathlib import Path
import collections,gzip,json,math,statistics
A=Path("artifacts/crow-v15-wing-normal-height-response-screen-20261112");B=Path("artifacts/crow-v15-wing-normal-heading-gain-screen-20261112");part="candidate-20261126-rebound-flexed/run.stderr.gz"
def read(p):
 states={};contacts=collections.defaultdict(list)
 with gzip.open(p,"rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);states[x["lane"],x["step"]]=x
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);
    if (e<2 and t<=1200) or (e>=6 and t<=600):contacts[e].append(tuple(z[1:]))
 return states,contacts
s,c=read(B/part);v,d=read(A/part);diff=[];count=nc=0
for e in (0,1,6,7):
 end=1200 if e<2 else 600
 for t in range(1,end+1):
  for k in ("q","v","policy_action","contacts"):
   if s[e,t][k]!=v[e,t][k]:diff.append([e,t,k])
  count+=1
 if c[e]!=d[e]:diff.append([e,"contacts"])
 nc+=len(c[e])
assert count==3600 and nc==57600,(count,nc)
out={"states":count,"contacts":nc,"differences":diff};(A/"parity.json").write_text(json.dumps(out,indent=2)+"\n");print("parity",out);assert not diff
meta=json.loads((A/"candidate-20261126-rebound-flexed/run.stdout").read_text())["first_episode_navigation"];rows=[]
def yaw(q):
 x,y,z,w=q[3:7];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
for e in range(2,8):
 end=min(meta["steps_by_environment"][e],1200);pre=[v[e,t] for t in range(551,601) if (e,t) in v];post=[v[e,t] for t in range(1001,end+1) if (e,t) in v];target=1.15 if e%2==0 else .90
 rows.append({"lane":e,"gain":[0,0,.6,.6,1.2,1.2,1.8,1.8][e],"target_m":target,"end_step":end,"pre_height_m":statistics.mean(x["q"][2] for x in pre) if pre else None,"post_height_m":statistics.mean(x["q"][2] for x in post) if post else None,"post_abs_height_error_m":statistics.mean(abs(x["q"][2]-target) for x in post) if post else None,"post_abs_heading_rad":statistics.mean(abs(yaw(x["q"])) for x in post) if post else None,"termination":meta["termination_reason_by_environment"][e]})
(A/"height-summary.json").write_text(json.dumps({"rows":rows,"scope":"Used-seed development screen; uncalibrated closure and command values; no flight qualification."},indent=2)+"\n")
for r in rows:print(r)
