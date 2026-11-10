from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-foot-placement-probe-20261110")
def frames(p):
 with gzip.open(p,"rt") as f:return json.load(f)["payload"]["frames"]
a=frames("artifacts/crow-v15-approach-i-early-recovery-20261110/development-i-approach-early-recovery.json.gz")
b=frames(A/"development-i-foot-placement.json.gz")
diff=[];count=0
for x,y in zip(a,b):
 if x["step"]>=2772:break
 count+=1
 for k in ("step","q","v","actor_observation","accepted_actions"):
  if x[k]!=y[k] and len(diff)<10:diff.append({"step":x["step"],"field":k})
r={"first_navigation_five_step_control":2772,"frames_before_navigation_five":count,"exact_equal":count==2771 and not diff,"differences":diff,"note":"Compare accepted states and policy actions before landing; not proof of parity of unlogged substep impulses."}
(A/"prefix-comparison.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
