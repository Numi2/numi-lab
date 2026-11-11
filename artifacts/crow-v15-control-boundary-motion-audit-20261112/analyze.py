from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-control-boundary-motion-audit-20261112");P=Path("artifacts/crow-v15-substep-gait-observer-20261111/autonomous-20261126/run.stderr.gz");q={}
with gzip.open(P,"rt") as f:
 for l in f:
  if l.startswith("CROW_Q_SUBSTEP "):
   z=l.split();t,k,e=map(int,z[1:4]);key=e,t,k;assert key not in q;q[key]=list(map(float,z[4:]))
assert len(q)==8*100*16
def stats(x):
 x=sorted(x);assert x
 return {"n":len(x),"mean_abs_rad_s2":sum(x)/len(x),"median_abs_rad_s2":x[len(x)//2],"p95_abs_rad_s2":x[int(.95*(len(x)-1))],"p99_abs_rad_s2":x[int(.99*(len(x)-1))],"max_abs_rad_s2":max(x)}
rows=[]
for e in range(8):
 for name,j in (("left_knee",15),("left_ankle",16),("right_knee",18),("right_ankle",19)):
  boundary=[];within=[];positive=negative=0
  for t in range(202,301):
   for k in range(16):
    def point(n):
     tt=t+n//16;kk=n%16
     return q[e,tt,kk][j]
    acc=(point(k)-2*point(k-1)+point(k-2))/(.00125**2)
    if k==0:
     boundary.append(abs(acc));positive+=acc>0;negative+=acc<0
    else:within.append(abs(acc))
  rows.append({"lane":e,"joint":name,"boundary":stats(boundary),"within":stats(within),"boundary_positive":positive,"boundary_negative":negative})
out={"rows":rows,"scope":"Second differences of logged q, baseline used development seed 20261126, steps 202-300; q sample timestep 1.25ms. Absolute distributions, not physical torque or biological stiffness."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+chr(10))
for j in ("left_knee","left_ankle","right_knee","right_ankle"):
 x=[r for r in rows if r["joint"]==j]; print(j,"boundary mean range",min(z["boundary"]["mean_abs_rad_s2"] for z in x),max(z["boundary"]["mean_abs_rad_s2"] for z in x),"within mean range",min(z["within"]["mean_abs_rad_s2"] for z in x),max(z["within"]["mean_abs_rad_s2"] for z in x),"boundary p95 range",min(z["boundary"]["p95_abs_rad_s2"] for z in x),max(z["boundary"]["p95_abs_rad_s2"] for z in x),"within p95 range",min(z["within"]["p95_abs_rad_s2"] for z in x),max(z["within"]["p95_abs_rad_s2"] for z in x))
