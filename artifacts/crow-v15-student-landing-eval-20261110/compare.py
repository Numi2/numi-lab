from pathlib import Path
import gzip,json,sys
A=Path("artifacts/crow-v15-student-landing-eval-20261110");O=Path("artifacts/crow-v15-stage-aware-eval-20261110");side=sys.argv[1];new=side+"-student-landing";old=side+"-stage-aware"
def frames(p):
 with gzip.open(p,"rt") as f:return json.load(f)["payload"]["frames"]
a=frames(O/(old+".json.gz"));b=frames(A/(new+".json.gz"));end=json.loads((O/(old+"-analysis.json")).read_text())["nav5_step"];differences=[];n=0
for x,y in zip(a,b):
 if x["step"]>=end:break
 n+=1
 for k in ("step","q","v","actor_observation","accepted_actions"):
  if x[k]!=y[k] and len(differences)<10:differences.append({"step":x["step"],"field":k})
r={"control_waypoint5_step":end,"matching_frames":n,"exact_equal_before_final_waypoint":n==end-1 and not differences,"differences":differences};(A/(new+"-prefix-comparison.json")).write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
