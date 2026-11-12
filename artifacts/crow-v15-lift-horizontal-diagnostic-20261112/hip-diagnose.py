from pathlib import Path
import gzip,json,os,subprocess,math,hashlib
p=Path("artifacts/crow-v15-lift-horizontal-diagnostic-20261112");src=Path("artifacts/crow-v15-small-support-stroke-20261112/candidate-20261127-rebound/run.stderr.gz");states={}
with gzip.open(src,"rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   z=json.loads(line.split(" ",1)[1])
   if z["lane"]==5 and z["step"] in range(225,301,5):states[z["step"]]=z
assert len(states)==16
req=[];mapping={}
for t,z in sorted(states.items()):
 for side,idx in (("left",15),("right",18)):
  for d in (0.,.2):
   n=len(mapping);mapping[n]=(t,side,d);q=z["q"].copy();q[idx-1]-=.5*d;q[idx]+=d;q[idx+1]-=.5*d;req.append(str(n)+" "+str(t)+" "+" ".join(map(str,q+z["v"])))
kin=Path("build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics");env=dict(os.environ,DYLD_LIBRARY_PATH=str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve()));r=subprocess.run([str(kin)],input="\n".join(req)+"\n",capture_output=True,text=True,check=True,env=env)
poses={}
for line in r.stdout.splitlines():
 z=line.split();n=int(z[1]);side=mapping[n][1]
 if z[0]=="body" and z[3]=="crow_"+side+"_foot":poses[n]=list(map(float,z[4:7]))
assert len(poses)==len(mapping)
rows=[]
for t in sorted(states):
 for side in ("left","right"):
  indices={d:n for n,(tt,ss,d) in mapping.items() if tt==t and ss==side};a=poses[indices[0.]];b=poses[indices[.2]]
  rows.append({"step":t,"side":side,"foot_center_delta_world_m":[b[i]-a[i] for i in range(3)]})
out={"rows":rows,"source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"kinematics_sha256":hashlib.sha256(kin.read_bytes()).hexdigest(),"scope":"Frozen torso and pose with hip -0.1, knee +0.2 and ankle -0.1 rad, foot box center world displacement. Not a simulated dynamic response or evidence that loaded contacts can remain fixed."};(p/"hip-diagnose.json").write_text(json.dumps(out,indent=2)+"\n")
for side in ("left","right"):
 x=[r["foot_center_delta_world_m"] for r in rows if r["side"]==side];print(side,"x range mm",min(v[0] for v in x)*1000,max(v[0] for v in x)*1000,"z range mm",min(v[2] for v in x)*1000,max(v[2] for v in x)*1000)
