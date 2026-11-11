from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-front-toe-long-replay-20261111");B=Path("artifacts/crow-v15-front-contact-regulation-20261111/results.txt.gz")
def read(p,lane):
 out={}
 with gzip.open(p,"rt") as f:
  for l in f:
   v=l.split()
   if v and v[0]=="trace" and int(v[1])==lane and int(v[3])<750:out[int(v[3])]=v[4:]
 return out
a=read(A/"results.txt.gz",1);b=read(B,14);diff=[t for t in range(750) if a.get(t)!=b.get(t)];report={"reference_lane":14,"samples":len(a),"differences":len(diff),"first_differences":diff[:20]};(A/"parity.json").write_text(json.dumps(report,indent=2)+"\n");print(report);assert len(a)==len(b)==750 and not diff
