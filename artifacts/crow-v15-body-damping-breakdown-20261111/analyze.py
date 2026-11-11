from pathlib import Path
import gzip,json,statistics
A=Path("artifacts/crow-v15-body-damping-breakdown-20261111");B=Path("artifacts/crow-v15-matched-balance-grip-screen-20261111/results.txt.gz")
def read(p):
 tr={};d={}
 with gzip.open(p,"rt") as f:
  for l in f:
   v=l.split()
   if not v:continue
   if v[0]=="trace":tr[int(v[3])]=v[4:]
   elif v[0]=="damping_body":d[int(v[1]),v[2]]=list(map(float,v[3:]))
 return tr,d
tr,d=read(A/"results.txt.gz");old,_=read(B);assert len(tr)==300 and all(tr.get(t)==old.get(t) for t in range(300));names=sorted({n for t,n in d});assert all((t,n) in d for t in range(300) for n in names)
out={"exact_trace_parity":True,"bodies":len(names),"windows":[]}
for lo,hi in ((20,70),(80,110),(120,160),(160,200),(200,240),(240,280)):
 rows=[]
 for n in names:
  v=[d[t,n] for t in range(lo,hi)];rows.append({"body":n,"linear_damping":v[0][0],"angular_damping":v[0][1],"mean_angular_moment":statistics.mean(x[2] for x in v),"mean_linear_moment":statistics.mean(x[3] for x in v),"mean_total_moment":statistics.mean(x[2]+x[3] for x in v),"mean_absolute_total_moment":statistics.mean(abs(x[2]+x[3]) for x in v)})
 out["windows"].append({"ticks":[lo,hi-1],"bodies":sorted(rows,key=lambda x:abs(x["mean_total_moment"]),reverse=True)})
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
