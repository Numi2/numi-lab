from pathlib import Path
import gzip,json,statistics
A=Path("artifacts/crow-v15-grip-moment-audit-20261111");B=Path("artifacts/crow-v15-matched-balance-grip-screen-20261111/results.txt.gz")
def read(p):
 tr={};w={};m={}
 with gzip.open(p,"rt") as f:
  for line in f:
   v=line.split()
   if not v:continue
   if v[0]=="trace":tr[int(v[3])]=v[4:]
   elif v[0]=="wrench":w[int(v[1])]=list(map(float,v[2:]))
   elif v[0]=="momentum":m[int(v[1])]=list(map(float,v[2:]))
 return tr,w,m
tr,w,m=read(A/"results.txt.gz");old=read(B)[0];diff=[t for t in range(300) if tr.get(t)!=old.get(t)]
assert len(tr)==len(w)==len(m)==300 and not diff,(len(tr),len(w),len(m),diff[:10])
out={"exact_trace_parity":True,"windows":[],"samples":[]}
for lo,hi in ((20,70),(80,110),(120,160),(160,200),(200,240),(240,280)):
 rows=[]
 for t in range(lo,hi):
  mass,cx,cy,cz,px,py,pz,hy=m[t];prev=m[t-1][-1];j=w[t];grav=mass*9.81*.02*(cx+m[t-1][1])/2;dh=hy-prev;rows.append((dh,j[3],grav,dh-j[3]-grav))
 out["windows"].append({"ticks":[lo,hi-1],"mean_delta_H_y":statistics.mean(x[0] for x in rows),"mean_contact_angular_impulse_y":statistics.mean(x[1] for x in rows),"mean_gravity_angular_impulse_y":statistics.mean(x[2] for x in rows),"mean_residual":statistics.mean(x[3] for x in rows),"mean_absolute_residual":statistics.mean(abs(x[3]) for x in rows)})
for t in (80,100,120,140,160,180,200,220,240,260,280):out["samples"].append({"tick":t,"momentum":m[t],"wrench":w[t]})
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
