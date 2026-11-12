from pathlib import Path
import collections,gzip,json,statistics
A=Path("artifacts/crow-v15-wing-normal-height-no-step-control-20261112");B=Path("artifacts/crow-v15-wing-normal-height-response-screen-20261112");part="candidate-20261126-rebound-flexed/run.stderr.gz"
def read(p):
 states={};contacts=collections.defaultdict(list)
 with gzip.open(p,"rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);states[x["lane"],x["step"]]=x
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);
    if t<=(1200 if e<2 else 600):contacts[e].append(tuple(z[1:]))
 return states,contacts
s,c=read(B/part);v,d=read(A/part);diff=[];count=nc=0
for e in range(8):
 end=1200 if e<2 else 600
 for t in range(1,end+1):
  for k in ("q","v","policy_action","contacts"):
   if s[e,t][k]!=v[e,t][k]:diff.append([e,t,k])
  count+=1
 if c[e]!=d[e]:diff.append([e,"contacts"])
 nc+=len(c[e])
assert count==6000 and nc==96000,(count,nc)
out={"states":count,"contacts":nc,"differences":diff};(A/"parity.json").write_text(json.dumps(out,indent=2)+"\n");print("parity",out);assert not diff
rows=[]
for e in range(2,8):
 def mean(k):return statistics.mean(s[e,t][k][2]-v[e,t][k][2] for t in range(1001,1201))
 rows.append({"lane":e,"step_direction":1 if e%2==0 else -1,"height_difference_m":mean("q"),"vertical_velocity_difference_m_s":mean("v"),"mean_abs_left_flap_angle_difference_rad":statistics.mean(abs(s[e,t]["q"][9]-v[e,t]["q"][9]) for t in range(601,1001)),"mean_abs_right_flap_angle_difference_rad":statistics.mean(abs(s[e,t]["q"][10]-v[e,t]["q"][10]) for t in range(601,1001))})
(A/"height-comparison.json").write_text(json.dumps({"rows":rows,"scope":"Matched used-seed simulation; does not validate aerodynamics or unseen conditions"},indent=2)+"\n")
for r in rows:print(r)
