from pathlib import Path
import gzip, json, os, subprocess
D=Path("artifacts/crow-v15-gait-phase-observer-20261111")
p=Path("artifacts/crow-v15-autonomous-gated-phase-walk-20261111/autonomous-20261126/run.stderr.gz")
with gzip.open(p,"rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   sample=json.loads(line.split(" ",1)[1])
   if sample["lane"]==3 and sample["step"]==330:break
 else:raise RuntimeError("missing source state")
rows=[]
for i in [-1,14,15,16,17,18,19]:
 q=sample["q"][:]
 if i>=0:q[i]+=.01
 rows.append("3 "+str(i+2)+" "+" ".join(map(str,q+sample["v"])))
env={**os.environ,"DYLD_LIBRARY_PATH":str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve())}
r=subprocess.run(["build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics"],input="\n".join(rows)+"\n",capture_output=True,text=True,env=env,check=True);body={}
names=("crow_left_foot","crow_right_foot","crow_left_front_toe_1_distal","crow_right_front_toe_1_distal")
for line in r.stdout.splitlines():
 x=line.split()
 if x[0]=="body" and x[3] in names:body[int(x[2]),x[3]]=float(x[6])
result={"source_lane":3,"source_step":330,"perturbation_rad":.01,"derivatives_m_per_rad":{str(i):{n:(body[i+2,n]-body[1,n])/.01 for n in names} for i in [14,15,16,17,18,19]},"limitation":"Local fixed-root kinematics at a single sampled pose; does not predict dynamic whole-body motion."}
(D/"kinematic-sensitivity.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
