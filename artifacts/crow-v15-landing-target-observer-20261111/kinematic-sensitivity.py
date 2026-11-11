from pathlib import Path
import gzip,hashlib,json,subprocess,os
R=Path.cwd();D=R/"artifacts/crow-v15-landing-target-observer-20261111";p=R/"artifacts/crow-v15-landing-staged-long-horizon-20261111/autonomous-20261126/replay.json.gz"
with gzip.open(p,"rt") as f:r=json.load(f)["payload"]
assert r["environment"]==5
x=next(x for x in r["frames"] if x["step"]==3705);q=x["q"];v=x["v"];tests=[("base",None,0.0),("left_plus",14,.02),("right_plus",17,.02),("left_minus",14,-.02),("right_minus",17,-.02)];lines=[]
for i,(_,j,delta) in enumerate(tests):
 y=q.copy()
 if j is not None:y[j]+=delta
 lines.append(str(i)+" 3705 "+" ".join(map(str,y+v)))
env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")};pout=subprocess.run([str(R/"build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics")],input="\n".join(lines)+"\n",text=True,capture_output=True,check=True,env=env).stdout;rows=[{"variant":a,"q_index":j,"offset_rad":delta} for a,j,delta in tests]
for line in pout.splitlines():
 z=line.split();i=int(z[1])
 if z[0]=="body" and z[3] in ("crow_left_foot","crow_right_foot"):rows[i][z[3]+"_position_m"]=[float(v) for v in z[4:7]]
 elif z[0]=="com":rows[i]["center_of_mass_m"]=[float(v) for v in z[4:7]]
out={"source_replay_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"replay_step":3705,"lane":5,"rows":rows,"scope":"Static finite differences at fixed root and all other coordinates. These do not model contact forces, resulting time evolution or control efficacy. Index and frame alignment need independent validation before mapping to applied drive targets."};(D/"kinematic-sensitivity.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
