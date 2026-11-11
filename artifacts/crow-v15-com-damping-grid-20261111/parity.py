from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-com-damping-grid-20261111");B=Path("artifacts/crow-v15-ankle-range-grid-20261111/results.txt.gz")
def read(p,lane):
 out={}
 with gzip.open(p,"rt") as f:
  for l in f:
   v=l.split()
   if v and v[0]=="trace" and int(v[1])==lane and int(v[3])<500:out[int(v[3])]=v[4:]
 return out
reports=[]
for current,reference in ((1,5),):
 a=read(A/"results.txt.gz",current);b=read(B,reference);diff=[t for t in range(500) if a.get(t)!=b.get(t)];reports.append({"current_lane":current,"reference_lane":reference,"samples":len(a),"differences":len(diff),"first_differences":diff[:20]});assert len(a)==len(b)==500 and not diff,reports
(A/"parity.json").write_text(json.dumps(reports,indent=2)+"\n");print(reports)
