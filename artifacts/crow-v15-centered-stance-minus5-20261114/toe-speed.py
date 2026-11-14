from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-centered-stance-minus5-20261114");values={0:[],1:[]}
with gzip.open(A/"trace.log.gz","rt") as f:
 for l in f:
  x=l.split()
  if x and x[0]=="step":
   e=int(x[1]);v=x.index("v");a=list(map(float,x[v+1:]));values[e].append(max(abs(a[j-1]) for j in range(23,39)))
out={str(e):{"samples":len(v),"maximum_accepted_toe_speed_rad_s":max(v,default=None),"threshold_12_rad_s_reached":any(t>=11.99 for t in v)} for e,v in values.items()};out["scope"]="accepted control states only; substep toe speed and effort not measured";(A/"toe-speed.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
