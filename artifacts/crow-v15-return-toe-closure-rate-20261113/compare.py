from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-return-toe-closure-rate-20261113");states={};front={}
with gzip.open(A/"trace.log.gz","rt") as f:
 for l in f:
  x=l.split()
  if not x:continue
  if x[0]=="step":
   q=x.index("q");v=x.index("v");states[int(x[1]),int(x[2])]=list(map(float,x[q+1:v]+x[v+1:]))
  elif x[0]=="control":front[int(x[1]),int(x[2])]=float(x[4])
on=next((t for t in range(450) if (0,t) in front and (1,t) in front and abs(front[0,t]-front[1,t])>1e-7),None);pre=[max(abs(a-b) for a,b in zip(states[0,t],states[1,t])) for t in range(on or 0) if (0,t) in states and (1,t) in states];out={"first_target_difference_zero_based":on,"prefix_pairs":len(pre),"prefix_exact":on is not None and len(pre)==on and max(pre,default=1)==0,"front_target_max":[max((front[e,t] for t in range(450) if (e,t) in front),default=None) for e in (0,1)]};(A/"comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
