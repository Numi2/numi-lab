from pathlib import Path
import gzip,itertools,json
r=Path("artifacts");base=r/"crow-v15-walk-hold-localized-route-20261111/autonomous-20261126/run.stderr.gz"; trial=r/"crow-v15-descent-damping-screen-20261111/autonomous-20261126/run.stderr.gz"
prior=json.loads((r/"crow-v15-full-landing-failure-analysis-20261111/analysis.json").read_text());limits={v["lane"]:next(t for t,p in v["phase_transitions"] if p==1.0) for v in prior["rows"]}
def stream(path):
 with gzip.open(path,"rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    a=json.loads(line.split(" ",1)[1]);lane=a["lane"];t=a["step"]
    if lane in limits and t<limits[lane]:yield (lane,t),[a[k] for k in ("q","v","policy_action","contacts")]
counts=[0]*8;mismatch=[]
for a,b in itertools.zip_longest(stream(base),stream(trial)):
 if a is None or b is None or a[0]!=b[0] or a[1]!=b[1]:
  mismatch.append({"reference":a[0] if a else None,"trial":b[0] if b else None});break
 counts[a[0][0]]+=1
out={"matched_q_v_actions_and_endpoint_contacts_by_lane":counts,"first_mismatch":mismatch,"comparison":"all recorded control endpoints strictly before reference phase-one entry; exact JSON-decoded values"}
(r/"crow-v15-descent-damping-screen-20261111/prefix-comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out))
