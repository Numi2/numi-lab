from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-contact-friction-screen-20261111");ref=Path("artifacts/crow-v15-hip-roll-staged-release-20261111/trace.log.gz");test=A/"control/trace.log.gz"
def rows(path):
 out={}
 with gzip.open(path,"rt") as f:
  for line in f:
   z=line.split()
   if z and z[0]=="step":
    q=z.index("q");v=z.index("v");out[int(z[1]),int(z[2])]=(list(map(float,z[q+1:v])),list(map(float,z[v+1:])))
 return out
r=rows(ref);c=rows(test);different=[];maximum=0.0
for k in sorted(set(r)&set(c)):
 for a,b in zip(r[k],c[k]):
  if len(a)!=len(b):different.append([*k,"length"]);continue
  delta=max((abs(x-y) for x,y in zip(a,b)),default=0.0);maximum=max(maximum,delta)
  if delta>0:different.append([*k,delta])
valid=len(r)==len(c)==4400 and r.keys()==c.keys() and not different
result={"reference_states":len(r),"control_states":len(c),"different_arrays":len(different),"max_abs_difference":maximum,"examples":different[:10],"parity":valid,"scope":"Exact decimal logged q/v parity; does not establish unlogged solver internals."};(A/"control-parity.json").write_text(json.dumps(result,indent=2)+"\n");print(result)
raise SystemExit(0 if valid else 1)
