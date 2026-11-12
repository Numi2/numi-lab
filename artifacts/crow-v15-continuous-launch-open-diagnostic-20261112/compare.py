from pathlib import Path
import collections,gzip,hashlib,json,sys
A=Path("artifacts/crow-v15-continuous-launch-open-diagnostic-20261112");B=Path("artifacts/crow-v15-continuous-leg-release-screen-20261112");seed=sys.argv[1];name="candidate-"+seed+"-rebound-hip"
def read(d):
 h={};n=collections.Counter()
 with gzip.open(d/name/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);t,e=x["step"],x["lane"];k="state"
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split(None,4);t,sub,e=map(int,z[1:4]);k="contact"
   else:continue
   if t<=625:
    key=e,t,k
    if key not in h:h[key]=hashlib.sha256()
    h[key].update(l.encode());n[key]+=1
 return {k:(v.hexdigest(),n[k]) for k,v in h.items()}
a,b=read(A),read(B);rows=[]
for e in range(8):
 x={"lane":e,"previous_arm":"same candidate"}
 for kind in ("state","contact"):
  start=1 if kind=="state" else 201; expected=1 if kind=="state" else 16
  missing=[t for t in range(start,626) if a.get((e,t,kind),(None,0))[1]!=expected or b.get((e,t,kind),(None,0))[1]!=expected]
  diffs=[t for t in range(start,626) if a.get((e,t,kind))!=b.get((e,t,kind))]
  x[kind+"_incomplete_steps"]=missing;x[kind+"_first_difference"]=min(diffs) if diffs else None
 rows.append(x)
(A/("paired-"+seed+".json")).write_text(json.dumps(rows,indent=2)+"\n")
for x in rows:print(x)
