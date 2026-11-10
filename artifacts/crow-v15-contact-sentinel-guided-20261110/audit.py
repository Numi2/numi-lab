from pathlib import Path
import gzip,json
A=Path("artifacts/crow-v15-contact-sentinel-guided-20261110");O=Path("artifacts/crow-v15-contact-release-hold-20261110");name="development-i-contact-sentinel-guided"
def frames(p):
 with gzip.open(p,"rt") as f:return json.load(f)["payload"]["frames"]
a=frames(O/"development-i-contact-release-hold.json.gz");b=frames(A/(name+".json.gz"));bad=[];n=0
for x,y in zip(a,b):
 if x["step"]>=2772:break
 n+=1
 for k in ("step","q","v","actor_observation","accepted_actions"):
  if x[k]!=y[k] and len(bad)<10:bad.append({"step":x["step"],"field":k})
nonfinite=[];last=0
with gzip.open(A/(name+".stderr.gz"),"rt") as f:
 for l in f:
  if l.startswith("CROW_TASK_STATE "):
   v=l.split();t=int(v[2])
   if t<last:break
   last=t
   if "nan" in l.lower() or "inf" in l.lower():nonfinite.append(t)
r={"equal_before_final_waypoint":n==2771 and not bad,"frames":n,"differences":bad,"nonfinite_task_steps":nonfinite,"note":"Only logged accepted task state checked; this is a guided development trial."};(A/"sentinel-audit.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
