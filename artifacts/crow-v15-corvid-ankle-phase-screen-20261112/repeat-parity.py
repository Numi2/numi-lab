from pathlib import Path
import gzip,json
a=Path("artifacts/crow-v15-corvid-stride-repeat-screen-20261112/candidate-20261127-rebound-fine/run.stderr.gz")
b=Path("artifacts/crow-v15-corvid-ankle-phase-screen-20261112/candidate-20261127-rebound-fine/run.stderr.gz")
def read(p):
 d={}
 with gzip.open(p,"rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);key=("s",x["lane"],x["step"],0);e=x["lane"];v=x
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);key=("c",e,t,k);v=z[4:]
   else:continue
   if e in (0,1,2,3):d[key]=v
 return d
x,y=map(read,(a,b));assert x.keys()==y.keys();diff=[key for key in x if x[key]!=y[key]]
out={"records":len(x),"states":sum(k[0]=="s" for k in x),"contacts":sum(k[0]=="c" for k in x),"differences":diff};Path("artifacts/crow-v15-corvid-ankle-phase-screen-20261112/repeat-parity.json").write_text(json.dumps(out,indent=2)+"\n");print(out);assert not diff
