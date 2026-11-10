from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-contact-release-hold-20261110")
def frames(p):
 with gzip.open(p,"rt") as f:return json.load(f)["payload"]["frames"]
r=json.loads((A/"release-audit.json").read_text());release=r.get("first_release_step")
if release is None: print("No event to compare");raise SystemExit(1)
a=frames("artifacts/crow-v15-foot-placement-probe-20261110/development-i-foot-placement.json.gz");b=frames(A/"development-i-contact-release-hold.json.gz")
diff=[];count=0
for x,y in zip(a,b):
 if x["step"]>=release: break
 count+=1
 for k in ("step","q","v","actor_observation","accepted_actions"):
  if x[k]!=y[k] and len(diff)<10:diff.append({"step":x["step"],"field":k})
out={"release_step":release,"frames_before_release":count,"exact_equal":count==release-1 and not diff,"differences":diff,"note":"Accepted endpoint parity only; unlogged substep impulses can differ."};(A/"prefix-comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
