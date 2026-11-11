from pathlib import Path
import gzip,importlib.util,json,math,os,subprocess
R=Path.cwd();A=Path("artifacts/crow-v15-takeoff-substep-wing-observer-20261112");P=A/"autonomous-20261126"
assert json.loads((A/"verification.json").read_text())["pass"]
s=importlib.util.spec_from_file_location("a","artifacts/crow-v15-whole-gait-audit-20261111/audit.py");a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
q={};contacts={}
with gzip.open(P/"run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_Q_SUBSTEP "):
   z=l.split();t,k,e=map(int,z[1:4]);assert (e,t,k) not in q;q[e,t,k]=list(map(float,z[4:]))
  elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=l.split();t,k,e=map(int,z[1:4])
   if 335<=t<=365:assert (e,t,k) not in contacts;contacts[e,t,k]=list(map(float,z[6:]))
assert len(q)==len(contacts)==8*31*16
inp="".join(f"{e} {t} {k} "+" ".join(map(str,v))+chr(10) for (e,t,k),v in sorted(q.items()));env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")}
p=subprocess.run(["build/crow-v15-takeoff-substep-wing-observer-20261112/kinematics"],input=inp,text=True,capture_output=True,env=env,check=True);body={}
for l in p.stdout.splitlines():
 z=l.split();key=int(z[1]),int(z[2]),int(z[3]),z[4];assert key not in body;body[key]=list(map(float,z[5:]))
assert len(body)==len(q)*3
with gzip.open(P/"replay.json.gz","rt") as f:r=json.load(f)["payload"]
assert r["environment"]==7
frames={x["step"]+1:x for x in r["frames"] if 334<=x["step"]<=364};assert len(frames)==31
poserr=roterr=0.
for t,f in frames.items():
 for n in ("crow_body","crow_left_wing","crow_right_wing"):
  x=body[7,t,15,n];i=r["body_names"].index(n)*13;y=f["body_states"][i:i+7]
  poserr=max(poserr,math.dist(x[:3],y[:3]));roterr=max(roterr,min(math.dist(x[3:7],y[3:7]),math.dist(x[3:7],[-v for v in y[3:7]])))
assert poserr<1e-5 and roterr<1e-5
def z(e,t,k,side):
 v=q[e,t,k];p=body[e,t,k,"crow_"+side+"_wing"];return a.rotate_inv(v[3:7],[p[i]-v[i] for i in range(3)])[2]
rows=[]
for e in range(8):
 loaded=[(t,k) for t in range(340,351) for k in range(16) if sum(contacts[e,t,k][2:4])>1e-7]
 last=loaded[-1]
 assert last[0]==348 and all(sum(contacts[e,t,k][2:4])<=1e-7 for t in (349,350) for k in range(16))
 rows.append({"lane":e,"last_loaded_substep":last,"torso_relative_wing_z_start":{side:z(e,347,15,side) for side in ("left","right")},"wing_z_change_to_last_loaded_m":{side:z(e,*last,side)-z(e,347,15,side) for side in ("left","right")},"wing_z_change_at_end_of_release_interval_m":{side:z(e,348,15,side)-z(e,347,15,side) for side in ("left","right")}})
out={"replay_fk_validation":{"max_position_error_m":poserr,"max_orientation_error":roterr},"rows":rows,"limits":"Modeled wing-body center displacement in torso coordinates is not a feather or aerodynamic-force measurement. Impulse threshold 1e-7 N s; coordinates and contact are after the same native substep but within-substep ordering remains unresolved. Development seed."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+chr(10));print(json.dumps(out,indent=2))
