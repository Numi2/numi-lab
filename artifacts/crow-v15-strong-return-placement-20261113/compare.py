from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-strong-return-placement-20261113");states={};placement={};gates={}
with gzip.open(A/"trace.log.gz","rt") as f:
 for line in f:
  x=line.split()
  if not x:continue
  if x[0]=="step":
   q=x.index("q");v=x.index("v");states[int(x[1]),int(x[2])]=list(map(float,x[q+1:v]+x[v+1:]))
  elif x[0]=="placement":placement[int(x[1]),int(x[2])]=float(x[3])
  elif x[0]=="placementGate":gates[int(x[1]),int(x[2])]=list(map(float,x[3:]))
on=next((t for t in range(450) if placement.get((1,t),0)>0),None);pre=[max(abs(a-b) for a,b in zip(states[0,t],states[1,t])) for t in range(on or 0) if (0,t) in states and (1,t) in states];c=[placement[1,t] for t in range(450) if (1,t) in placement];rates=[b-a for a,b in zip(c,c[1:])];data={"first_placement_zero_based":on,"prefix_pairs":len(pre),"prefix_exact":on is not None and len(pre)==on and max(pre,default=1)==0,"maximum_progress":max(c,default=None),"max_step":max(rates,default=None),"bounds_valid":bool(c) and min(c)>=0 and max(c)<=1.000001 and max(rates,default=1)<=.050001 and min(rates,default=-1)>=-1e-7};(A/"comparison.json").write_text(json.dumps(data,indent=2)+"\n");print(json.dumps(data,indent=2))
