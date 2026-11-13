from pathlib import Path
import gzip,json,hashlib
A=Path("artifacts/crow-v15-hop-wing-transition-20261113"); arms={}
for arm in ("push","hold"):
 p=A/(arm+"-wing-20261126-rebound-transition/run.stderr.gz"); d={}
 with gzip.open(p,"rt") as f:
  for line in f:
   z=line.split()
   if z and z[0]=="CROW_WING_TARGET_PAIR":
    t,e=map(int,z[1:3]);d[e,t]=list(map(float,z[3:]))
 arms[arm]=d
rows=[]
for e in range(2,8):
 missing=[t for t in range(348,401) if any((e,t) not in arms[a] for a in arms)]
 differences=[t for t in range(348,401) if (e,t) in arms["push"] and (e,t) in arms["hold"] and arms["push"][e,t]!=arms["hold"][e,t]]
 rows.append({"lane":e,"missing_ticks":missing,"different_wing_target_ticks":differences})
result={"valid":all(not x["missing_ticks"] and not x["different_wing_target_ticks"] for x in rows),"rows":rows,"scope":"The target-pair observer checks the two flap target histories through tick 400, not physical wing equality or other wing joints."}
(A/"matched-wing-check.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
if not result["valid"]:raise SystemExit(1)
