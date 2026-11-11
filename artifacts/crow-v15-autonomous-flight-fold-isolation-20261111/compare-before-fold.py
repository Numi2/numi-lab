from pathlib import Path
import gzip,json
ref=Path("artifacts/crow-v15-matched-release-inhibited-20261111/autonomous-20261126/run.stderr.gz")
cur=Path("artifacts/crow-v15-autonomous-flight-fold-isolation-20261111/autonomous-20261126/run.stderr.gz")
def read(p):
 out={}
 with gzip.open(p,"rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1])
    if x["step"]<=369:out[x["lane"],x["step"]]=x
 return out
a,b=read(ref),read(cur);rows=[]
for e in range(8):
 keys=[(e,t) for t in range(1,370)]
 missing=[t for _,t in keys if (e,t) not in a or (e,t) not in b]
 different=[t for _,t in keys if (e,t) in a and (e,t) in b and any(a[e,t][k]!=b[e,t][k] for k in ("q","v","policy_action","contacts"))]
 rows.append({"lane":e,"missing":len(missing),"differing":len(different),"first_difference":different[:1]})
out={"rows":rows,"scope":"Sampled accepted endpoints before the earliest possible fold; exact q, velocity, policy output and reported contacts. This verifies only the compared interval and does not by itself establish self-contact-observer completeness."}
Path("artifacts/crow-v15-autonomous-flight-fold-isolation-20261111/autonomous-20261126/comparison-before-fold.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
