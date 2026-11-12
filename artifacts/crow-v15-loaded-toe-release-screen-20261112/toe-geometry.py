from pathlib import Path
import gzip,json,subprocess,os,math,hashlib
p=Path("artifacts/crow-v15-supporting-leg-stroke-20261112/sham-20261127-stance/run.stderr.gz");state=None
with gzip.open(p,"rt") as f:
 for l in f:
  if l.startswith("CROW_SUPPORT_SAMPLE "):
   z=json.loads(l.split(" ",1)[1]);
   if z["lane"]==5 and z["step"]==305:state=z;break
assert state;q=state["q"];v=state["v"];lines=[];offsets=(-.2,-.1,0,.1)
for i,d in enumerate(offsets):
 qq=q.copy();qq[29]+=d;qq[30]+=.75*d;lines.append(str(i)+" 305 "+" ".join(map(str,qq+v)))
bin=Path("build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics");env=dict(os.environ,DYLD_LIBRARY_PATH=os.path.abspath("build/crow-v15-hip-roll-journey-integration-20261111"));r=subprocess.run([str(bin)],input="\n".join(lines)+"\n",capture_output=True,text=True,check=True,env=env);old=json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text());ground=next(x["ground_z_m"] for x in old["rows"] if x["seed"]==20261127);rows=[]
for l in r.stdout.splitlines():
 z=l.split()
 if z[0]=="body" and z[3]=="crow_left_rear_toe_distal":
  v=list(map(float,z[4:]));x,y,wz,w=v[3:7];rot=(2*(x*wz-y*w),2*(y*wz+x*w),1-2*(x*x+y*y));clear=v[2]-ground-sum(abs(j)*k for j,k in zip(rot,(.009,.0025,.0092)));rows.append({"proximal_offset_rad":offsets[int(z[1])],"distal_offset_rad":offsets[int(z[1])]*.75,"clearance_m":clear})
assert len(rows)==4 and rows[1]["clearance_m"]>rows[2]["clearance_m"]
out={"source":"supporting-leg stroke lane 5 step 305", "fk_sha256":hashlib.sha256(bin.read_bytes()).hexdigest(),"rows":rows,"scope":"Frozen-pose kinematics only; offsets from current joint states, not joint targets, forces, reachability or in-vivo data."};Path("artifacts/crow-v15-loaded-toe-release-screen-20261112/toe-geometry.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
