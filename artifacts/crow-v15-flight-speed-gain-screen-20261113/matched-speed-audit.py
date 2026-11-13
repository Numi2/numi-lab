from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-flight-speed-gain-screen-20261113");P=A/"push-wing-20261126-rebound-gain";Q=Path("artifacts/crow-v15-flight-speed-feedback-20261113/push-wing-20261126-rebound-speed");data=[]
for p in (Q,P):
 d={"state":{},"target":{}};data.append(d)
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);d["state"][x["lane"],x["step"]]=x
   elif line.startswith("CROW_DRIVE_TARGET "):
    z=line.split();t,e=map(int,z[1:3]);d["target"][e,t]=list(map(float,z[3:]))
old,new=data;rows=[]
for e in range(2,8):
 first=next((t for t in range(335,901) if (e,t) in old["target"] and (e,t) in new["target"] and old["target"][e,t][6:8]!=new["target"][e,t][6:8]),None);missing=[];diff={x:[] for x in ("q","v","policy_action","contacts")}
 if first is None:missing.append("no target difference")
 else:
  for t in range(1,first):
   if (e,t) not in old["state"] or (e,t) not in new["state"]:missing.append(t);continue
   for x in diff:
    if old["state"][e,t][x]!=new["state"][e,t][x]:diff[x].append(t)
 rows.append({"lane":e,"first_target_difference":first,"missing_ticks":missing,"differences":diff,"parity_before_first_target_difference":bool(first is not None and not missing and not any(diff.values()))})
(A/"matched-speed-audit.json").write_text(json.dumps({"rows":rows},indent=2)+"\n");print(json.dumps(rows,indent=2))
