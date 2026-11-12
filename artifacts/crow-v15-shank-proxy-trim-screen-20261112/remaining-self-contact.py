from pathlib import Path
import gzip,json,collections
p=Path("artifacts/crow-v15-shank-proxy-trim-screen-20261112");seen=set();first={};pairs=collections.Counter()
with gzip.open(p/"candidate-20261127/run.stderr.gz","rt") as h:
 for l in h:
  if not l.startswith("CROW_SUPPORT_SAMPLE "):continue
  x=json.loads(l.split(" ",1)[1]);e,t=x["lane"],x["step"]
  if not 1<=t<=100:continue
  for c in x["contacts"]:
   pair=(c["a_name"],c["b_name"])
   if c["impulses"][0]>1e-7 and any("shank" in q for q in pair) and any("rear_toe_distal" in q for q in pair):seen.add((e,t));first.setdefault(e,t);pairs[tuple(sorted(pair))]+=1
out={"lane_endpoints_with_remaining_distal_shank_contact":len(seen),"possible_lane_endpoints":800,"first_step_by_lane":first,"pair_contact_counts":[{"pair":k,"count":v} for k,v in pairs.items()],"scope":"Endpoint positive-impulse samples during first 100 intervals only, not a measure of anatomical penetration or substep coverage."};(p/"remaining-self-contact.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
