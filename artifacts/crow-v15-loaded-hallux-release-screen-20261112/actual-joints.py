from pathlib import Path
import gzip,json,math,statistics
p=Path("artifacts/crow-v15-loaded-hallux-release-screen-20261112")
def load(name):
 out={}
 with gzip.open(p/name/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);t=x["step"];e=x["lane"]
    if 201<=t<=300:out[e,t]=x["q"]
 assert len(out)==800
 return out
s=load("sham-20261127-paired");c=load("candidate-20261127-rebound");rows=[]
for side,j,sign in (("left",15,1),("right",18,-1)):
 vals=[]
 for e in range(8):
  for t in range(201,301):
   carrier=math.sin(2*math.pi*(t*.02+.005)/.50)*sign
   if carrier>.7:vals.append((c[e,t][j]-s[e,t][j],c[e,t][j+1]-s[e,t][j+1],c[e,t][j-1]-s[e,t][j-1]))
 rows.append({"side":side,"samples":len(vals),"median_hip_difference_rad":statistics.median(h for x,y,h in vals),"median_knee_difference_rad":statistics.median(x for x,y,h in vals),"median_ankle_difference_rad":statistics.median(y for x,y,h in vals),"min_knee_difference_rad":min(x for x,y,h in vals),"max_knee_difference_rad":max(x for x,y,h in vals)})
out={"rows":rows,"scope":"Endpoint joint-angle differences against matched unloaded-toe-release sham at carrier > 0.7; trajectories diverge and include indirect dynamic effects. These are not isolated actuator tracking errors or in-vivo kinematics."}
(p/"actual-joints.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
