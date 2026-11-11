from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-aft-com-grip-grid-20261111");B=Path("artifacts/crow-v15-aft-com-balance-replay-20261111/results.txt.gz")
def read(p):
 out={}
 with gzip.open(p,"rt") as f:
  for l in f:
   v=l.split()
   if v and v[0]=="trace" and int(v[1])==1 and int(v[3])<500:out[int(v[3])]=v[4:]
 return out
a=read(A/"results.txt.gz");b=read(B);diff=[t for t in range(500) if a.get(t)!=b.get(t)];r={"reference_samples":len(a),"differences":len(diff),"first_differences":diff[:20]};(A/"parity.json").write_text(json.dumps(r,indent=2)+"\n");print(r);assert len(a)==len(b)==500 and not diff
