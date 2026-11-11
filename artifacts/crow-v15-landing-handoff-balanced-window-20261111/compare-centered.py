from pathlib import Path
import gzip,json
P=Path("artifacts/crow-v15-landing-handoff-balanced-window-20261111/autonomous-20261126")
paths=[Path("artifacts/crow-v15-landing-handoff-center-control-20261111/autonomous-20261126/run.stderr.gz"),P/"run.stderr.gz"]
def read(p):
 out={}
 with gzip.open(p,"rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1])
    if x["lane"]==7 and 3000<=x["step"]<=3801:out[x["step"]]=x
 return out
a,b=map(read,paths);common=sorted(set(a)&set(b));different=[t for t in common if any(a[t][k]!=b[t][k] for k in ("q","v","policy_action","contacts"))]
o={"lane":7,"compared":len(common),"first_step":common[0],"last_step":common[-1],"differing":len(different),"first_difference":different[:1],"scope":"Accepted endpoint telemetry comparison only; state indices have a one-step offset relative to replay frames."}
(P/"comparison-centered.json").write_text(json.dumps(o,indent=2)+"\n")
print(json.dumps(o,indent=2))
