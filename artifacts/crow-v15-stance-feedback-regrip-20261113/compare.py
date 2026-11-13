from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-stance-feedback-regrip-20261113");states={};feedback={};command={}
with gzip.open(A/"trace.log.gz","rt") as f:
 for line in f:
  x=line.split()
  if not x:continue
  if x[0]=="step":
   q=x.index("q");v=x.index("v");states[int(x[1]),int(x[2])]=list(map(float,x[q+1:v]+x[v+1:]))
  elif x[0]=="hipfeedback":feedback[int(x[1]),int(x[2])]=list(map(float,x[3:]))
  elif x[0]=="rightHipCommand":command[int(x[1]),int(x[2])]=float(x[3])
on=next((t for t in range(450) if abs(feedback.get((1,t),[0])[0])>1e-8),None);pre=[max(abs(a-b) for a,b in zip(states[0,t],states[1,t])) for t in range(on or 0) if (0,t) in states and (1,t) in states];c=[feedback[1,t][0] for t in range(450) if (1,t) in feedback];rates=[abs(b-a) for a,b in zip(c,c[1:])];data={"first_correction_zero_based":on,"prefix_pairs":len(pre),"prefix_exact":on is not None and len(pre)==on and max(pre,default=1)==0,"max_abs_correction_rad":max(map(abs,c),default=None),"max_correction_step_rad":max(rates,default=None),"bounds_valid":bool(c) and max(map(abs,c))<=.003001 and max(rates,default=1)<=.000501,"feedback":{str(t):feedback[1,t] for t in range(450) if (1,t) in feedback}};(A/"comparison.json").write_text(json.dumps(data,indent=2)+"\n");print(json.dumps({k:v for k,v in data.items() if k!="feedback"},indent=2))
