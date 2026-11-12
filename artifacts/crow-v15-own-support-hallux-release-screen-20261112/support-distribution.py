from pathlib import Path
import gzip,json,statistics,collections
p=Path("artifacts/crow-v15-earlier-hallux-release-screen-20261112");out=Path("artifacts/crow-v15-own-support-hallux-release-screen-20261112");es=json.loads((p/"target-timing.json").read_text())["earlier_phase_candidate"]["events"];d={}
with gzip.open(p/"candidate-20261127-rebound/run.stderr.gz","rt") as f:
 for l in f:
  z=l.split()
  if z and z[0]=="CROW_TWO_TOE_SUBSTEP":
   t,k,e=map(int,z[1:4]);d[e,t,k]=list(map(float,z[6:]))
result={}
for delta in (3,2,1,0):
 c=collections.Counter();rat=[]
 for x in es:
  e=x["lane"];side=x["side"];t=x["last_front"][0]-delta;ds=[d[e,t,k] for k in range(16)];own=sum(z[2+side] for z in ds);front=sum(sum(z[5+8*side:11+8*side]) for z in ds);rear=sum(sum(z[11+8*side:13+8*side]) for z in ds);other=own-rear
  c["own_positive"]+=own>1e-7;c["front_positive"]+=front>1e-7;c["nonrear_positive"]+=other>1e-7;rat.append(other/own if own>0 else 0)
 result[str(delta)]={"counts":dict(c),"nonrear_fraction_min":min(rat),"nonrear_fraction_median":statistics.median(rat)}
result["method"]="Sums normal impulse over each 20ms interval and all grouped toe segments in the preceding failed matched candidate. Total own foot includes rear; subtracting rear is a nonrear load estimate, not a per-substep or future support guarantee."
(out/"support-distribution.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
