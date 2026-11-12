from pathlib import Path
import collections,gzip,hashlib,json
A=Path("artifacts/crow-v15-active-contact-failure-observer-20261112");B=Path("artifacts/crow-v15-continuous-launch-route-diagnostic-20261112");name="candidate-20261126-rebound-hip"
def read(d):
 meta=json.loads((d/name/"run.stdout").read_text())["first_episode_navigation"];ends=meta["steps_by_environment"];h={};n=collections.Counter()
 with gzip.open(d/name/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);t,e=x["step"],x["lane"];k="state"
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split(None,4);t,sub,e=map(int,z[1:4]);k="contact"
   else:continue
   if t<=ends[e]:
    key=e,t,k
    if key not in h:h[key]=hashlib.sha256()
    h[key].update(l.encode());n[key]+=1
 return {k:(v.hexdigest(),n[k]) for k,v in h.items()},ends
a,ae=read(A);b,be=read(B);rows=[]
for e in range(8):
 end=min(ae[e],be[e]);x={"lane":e,"common_first_episode_end":end}
 for kind in ("state","contact"):
  start=1 if kind=="state" else 201;expected=1 if kind=="state" else 16
  missing=[t for t in range(start,end+1) if a.get((e,t,kind),(None,0))[1]!=expected or b.get((e,t,kind),(None,0))[1]!=expected]
  diffs=[t for t in range(start,end+1) if a.get((e,t,kind))!=b.get((e,t,kind))]
  x[kind+"_incomplete_steps"]=missing;x[kind+"_first_difference"]=min(diffs) if diffs else None
 rows.append(x)
(A/"first-episode-parity.json").write_text(json.dumps(rows,indent=2)+"\n")
for x in rows:print(x)
