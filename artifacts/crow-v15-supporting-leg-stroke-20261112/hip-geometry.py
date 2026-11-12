from pathlib import Path
import gzip,json,subprocess,os,hashlib
p=Path("artifacts/crow-v15-upper-body-transition-isolation-20261112/sham-20261127-held/run.stderr.gz");state=None;impulses=[]
with gzip.open(p,"rt") as f:
 for l in f:
  if l.startswith("CROW_SUPPORT_SAMPLE "):
   z=json.loads(l.split(" ",1)[1])
   if z["lane"]==5 and z["step"]==320:state=z
  elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=l.split();t,k,e=map(int,z[1:4])
   if e==5 and t==320:impulses.append((k,list(map(float,z[6:]))))
assert state and len(impulses)==16
normal=[sum(v[2+i] for k,v in impulses) for i in range(2)]
q=state["q"];v=state["v"];lines=[]
for i,d in enumerate((-.1,0,.1)):
 qq=q.copy();qq[14]+=d;lines.append(str(i)+" 320 "+" ".join(map(str,qq+v)))
bin=Path("build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics");env=dict(os.environ,DYLD_LIBRARY_PATH=os.path.abspath("build/crow-v15-hip-roll-journey-integration-20261111"));run=subprocess.run([str(bin)],input="\n".join(lines)+"\n",capture_output=True,text=True,check=True,env=env);rows=[]
for l in run.stdout.splitlines():
 z=l.split()
 if z[0]=="body" and z[3] in ("crow_left_foot","crow_left_front_toe_2_distal"):rows.append({"offset_rad":[-.1,0,.1][int(z[1])],"body":z[3],"position_m":list(map(float,z[4:7]))})
assert len(rows)==6 and normal[0]>0
out={"source":"retained held sham lane 5 step 320", "impulse_left_right":normal,"kinematics_sha256":hashlib.sha256(bin.read_bytes()).hexdigest(),"positions":rows,"scope":"Frozen-pose native forward kinematics, torso held fixed, not a dynamic step or force prediction; hip angle offsets are diagnostic."}
Path("artifacts/crow-v15-supporting-leg-stroke-20261112/hip-geometry.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
