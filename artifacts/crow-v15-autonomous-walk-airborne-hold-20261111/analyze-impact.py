from pathlib import Path
import collections,gzip,json
D=Path("artifacts/crow-v15-autonomous-walk-airborne-hold-20261111")
paths={"baseline":Path("artifacts/crow-v15-autonomous-sustained-walk-control-20261111/autonomous-20261126/run.stderr.gz"),"hold":D/"autonomous-20261126/run.stderr.gz"};out={}
for label,p in paths.items():
 peaks={};cap=collections.Counter();maxspeed=collections.defaultdict(float);coverage=collections.Counter()
 with gzip.open(p,"rt") as f:
  for l in f:
   if l.startswith("CROW_TWO_TOE_SUBSTEP "):
    x=l.split();t,k,e=map(int,x[1:4])
    if 210<=t<=300:
     v=float(x[6])+float(x[7]);coverage[e,t,k]+=1
     if e not in peaks or v>peaks[e][0]:peaks[e]=(v,t,k)
   elif l.startswith("CROW_TOE_V_SUBSTEP "):
    x=l.split();t,k,e=map(int,x[1:4])
    if 201<=t<=600:
     v=max(abs(float(y)) for y in x[4:]);cap[e]+=int(v>=11.99);maxspeed[e]=max(maxspeed[e],v)
 assert len(coverage)==8*91*16 and all(v==1 for v in coverage.values())
 out[label]={"peak_upward_foot_impulse_Ns":{str(e):{"value":v[0],"step":v[1],"substep":v[2]} for e,v in sorted(peaks.items())},"toe_speed_cap_substeps":dict(cap),"maximum_toe_joint_speed_rad_s":dict(maxspeed)}
out["limitations"]="One used development seed/layout; association is not a causality or biological-realism finding. Cap threshold is 11.99 rad/s.";(D/"impact-analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
