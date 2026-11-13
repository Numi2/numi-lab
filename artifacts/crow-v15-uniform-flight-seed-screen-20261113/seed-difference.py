from pathlib import Path
import gzip,json,sys
A=Path("artifacts/crow-v15-uniform-flight-seed-screen-20261113");variant=sys.argv[1];paths=[A/("push-wing-"+variant+"-"+str(seed)+"-rebound-uniform") for seed in (20261126,20261127)];states=[]
for p in paths:
 d={};states.append(d)
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for line in f:
   if not line.startswith("CROW_SUPPORT_SAMPLE "):continue
   x=json.loads(line.split(" ",1)[1]);t=x["step"];e=x["lane"]
   if e>=2 and t in (1,100,300,500,700,900):d[e,t]={k:x[k] for k in ("q","v","policy_action","contacts")}
rows=[]
for e in range(2,8):
 comparison=[]
 for t in (1,100,300,500,700,900):
  a=states[0].get((e,t));b=states[1].get((e,t));comparison.append({"tick":t,"both_present":a is not None and b is not None,"exact_state_match":a==b if a is not None and b is not None else None})
 rows.append({"lane":e,"samples":comparison})
out={"variant":variant,"seeds":[20261126,20261127],"rows":rows,"scope":"Compare sampled accepted states only, not independence or full-distribution coverage."};(A/("seed-difference-"+variant+".json")).write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
