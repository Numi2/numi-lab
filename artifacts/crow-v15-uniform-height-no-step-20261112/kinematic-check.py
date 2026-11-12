from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-uniform-height-no-step-20261112")
P=Path("artifacts/crow-v15-uniform-height-screen-20261112/candidate-20261126-rebound-hip/run.stderr.gz")
d={}
with gzip.open(P,"rt") as f:
 for l in f:
  if l.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(l.split(" ",1)[1]);s=x["step"]-1;e=x["lane"]
   if 598<=s<=700 and e in (0,4,6):d[s,e]=(x["q"][2],x["v"][2])
out=[]
for e in (4,6):
 errors=[];rates=[]
 for t in range(599,701):
  za,va=d[t,e];zb,vb=d[t,0];pa,pva=d[t-1,e];pb,pvb=d[t-1,0]
  fd=((za-zb)-(pa-pb))/.02;approx=.5*((va-vb)+(pva-pvb));errors.append(fd-approx);rates.append(fd)
 out.append({"lane":e,"control_lane":0,"mean_abs_derivative_discrepancy_m_s":sum(map(abs,errors))/len(errors),"max_abs_derivative_discrepancy_m_s":max(map(abs,errors)),"peak_abs_differential_rate_m_s":max(map(abs,rates)),"scope":"50 Hz endpoints, trapezoidal velocity approximation, different related starts; does not identify force sources or exclude assistance"})
(A/"kinematic-check.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
