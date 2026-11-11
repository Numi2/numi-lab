from pathlib import Path
import gzip,json,os,subprocess
A=Path("artifacts/crow-v15-staged-weight-shift-gait-screen-20261111");source=A/"autonomous-20261126/run.stderr.gz";sample=None
with gzip.open(source,"rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1])
   if x["lane"]==5 and x["step"]==201:sample=x;break
assert sample is not None
q=sample["q"];v=sample["v"];lines=[]
for e,j in enumerate((None,14,15,16)):
 y=q.copy()
 if j is not None:y[j]+=.05
 lines.append(str(e)+" 1 "+" ".join(map(str,y+v))+"\n")
env={**os.environ,"DYLD_LIBRARY_PATH":str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve())};out=subprocess.run(["build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics"],input="".join(lines),text=True,capture_output=True,check=True,env=env).stdout;rows={}
for line in out.splitlines():
 a=line.split()
 if a[0]=="body" and a[3] in ("crow_left_foot","crow_left_front_toe_1_distal","crow_left_rear_toe_distal"):rows.setdefault(a[3],{})[int(a[1])]=list(map(float,a[4:7]))
r={"sample":{"seed":20261126,"lane":5,"observer_step":201},"increment_rad":.05,"perturbed_q_index":{"1":14,"2":15,"3":16},"positions_m":rows,"limits":"Instantaneous fixed-root forward kinematics at one observed configuration. Not a dynamic contact prediction, biological calibration, or end-to-end walking outcome."};(A/"kinematic-sensitivity.json").write_text(json.dumps(r,indent=2)+"\n")
