from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-flight-negative-sweep-20261113");P=A/"push-wing-20261126-rebound-negative"
with gzip.open(A/"sweep-reference.json.gz","rt") as f: ref=json.load(f)
meta=json.loads((P/"run.stdout").read_text())["first_episode_navigation"];states={};dups=[]
with gzip.open(P/"run.stderr.gz","rt") as f:
 for line in f:
  if not line.startswith("CROW_SUPPORT_SAMPLE "):continue
  x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
  if e not in (2,3) or t>ref["ends"][e]:continue
  key=f"{e},{t}";
  if key in states:dups.append(key)
  states[key]={n:x[n] for n in ("q","v","policy_action","contacts")}
rows=[]
for e in (2,3):
 end=ref["ends"][e];missing=[];diff=[]
 for t in range(1,end+1):
  k=f"{e},{t}"
  if k not in states or k not in ref["states"]:missing.append(t)
  elif states[k]!=ref["states"][k]:diff.append(t)
 rows.append({"lane":e,"reference_end":end,"new_end":meta["steps_by_environment"][e],"missing_ticks":missing,"different_ticks":diff,"full_first_episode_parity":bool(end==meta["steps_by_environment"][e] and not missing and not diff and not dups)})
obj={"reference_commit":ref["source_commit"],"reference_log_sha256":ref["source_log_sha256"],"rows":rows,"duplicates":dups};(A/"repeated-sweep-audit.json").write_text(json.dumps(obj,indent=2)+"\n");print(json.dumps(obj,indent=2))
