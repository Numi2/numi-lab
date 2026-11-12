from pathlib import Path
import collections,gzip,hashlib,json
A=Path("artifacts/crow-v15-launch-wing-ramp-screen-20261112");B=Path("artifacts/crow-v15-coordinated-launch-handoff-20261112");name="candidate-20261126-rebound-hip"
def read(d):
 h={};n=collections.Counter()
 with gzip.open(d/name/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);t,e=x["step"],x["lane"];k="state"
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split(None,4);t,sub,e=map(int,z[1:4]);k="contact"
   else:continue
   if t<=575:
    key=e,t,k
    if key not in h:h[key]=hashlib.sha256()
    h[key].update(l.encode());n[key]+=1
 return {k:(v.hexdigest(),n[k]) for k,v in h.items()}
a,b=read(A),read(B);r=[]
for e in range(8):
 x={"lane":e}
 for kind in ("state","contact"):
  diffs=[t for t in range(1,576) if a.get((e,t,kind))!=b.get((e,t,kind))];x[kind+"_first_difference"]=min(diffs) if diffs else None
 r.append(x)
(A/"coordinated-parity.json").write_text(json.dumps(r,indent=2)+"\n")
for x in r:print(x)
