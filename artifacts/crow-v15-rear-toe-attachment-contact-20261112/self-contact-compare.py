from pathlib import Path
import gzip,json,collections
p=Path("artifacts/crow-v15-rear-toe-attachment-contact-20261112");out={}
for n in ("control-20261127","candidate-20261127"):
 first={};endpoints=set()
 with gzip.open(p/n/"run.stderr.gz","rt") as f:
  for l in f:
   if not l.startswith("CROW_SUPPORT_SAMPLE "):continue
   x=json.loads(l.split(" ",1)[1]);e,t=x["lane"],x["step"]
   if not 1<=t<=100:continue
   for c in x["contacts"]:
    pair=(c["a_name"],c["b_name"])
    if c["impulses"][0]>1e-7 and any("shank" in q for q in pair) and any("rear_toe" in q for q in pair):first.setdefault(str(e),t);endpoints.add((e,t))
 out[n]={"first_by_lane":first,"positive_self_contact_lane_endpoints":len(endpoints),"available_lane_endpoints":800}
out["scope"]="Endpoint contacts, not all substeps; compares first 100 intervals before any recorded first-episode termination. Initial perturbations differ physically after geometry switch. Positive impulse threshold 1e-7.";(p/"self-contact-compare.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
