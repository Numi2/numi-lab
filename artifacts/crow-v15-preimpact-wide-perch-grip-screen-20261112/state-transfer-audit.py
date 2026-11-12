from pathlib import Path
import gzip,json,subprocess,os,math
p=Path("artifacts/crow-v15-preimpact-wide-perch-grip-screen-20261112");meta=json.loads((p/"source-states.json").read_text());
with gzip.open(meta["source"],"rt") as f:r=json.load(f)["payload"]
frames={x["step"]:x for x in r["frames"]};pi=r["body_names"].index("crow_course_perch")*13;items=[]
for e,row in enumerate(meta["selection"]):
 f=frames[row["step"]];q=f["q"][:];v=f["v"][:];c=f["body_states"][pi:pi+3];q[0]-=c[0];q[1]-=c[1];q[2]-=c[2]+.055;items.append((e,q,v,c))
text="".join(str(e)+" 1 "+" ".join(map(str,q+v))+"\n" for e,q,v,_ in items)
env={**os.environ,"DYLD_LIBRARY_PATH":str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve())}
ret=subprocess.run(["build/crow-v15-gait-slip-observer-20261112/kinematics"],input=text,text=True,capture_output=True,env=env,check=True)
calc={}
for line in ret.stdout.splitlines():
 z=line.split()
 if z and z[0]=="body":calc[int(z[1]),z[3]]=list(map(float,z[4:]))
rows=[]
for e,q,v,c in items:
 f=frames[meta["selection"][e]["step"]];worst={"position_m":0.,"orientation_component":0.,"velocity_m_s":0.,"angular_velocity_rad_s":0.};details=[]
 for name in r["body_names"][:35]:
  z=calc[e,name];idx=r["body_names"].index(name)*13;raw=f["body_states"][idx:idx+13];position=[raw[0]-c[0],raw[1]-c[1],raw[2]-c[2]-.055];errs=[math.dist(z[:3],position),min(max(abs(z[i+3]-raw[i+3]) for i in range(4)),max(abs(z[i+3]+raw[i+3]) for i in range(4))),math.dist(z[7:10],raw[7:10]),math.dist(z[10:13],raw[10:13])];
  for key,val in zip(worst,errs):worst[key]=max(worst[key],val)
  if name in ("crow_left_foot","crow_right_foot","crow_left_front_toe_2_distal","crow_right_front_toe_2_distal"):details.append({"body":name,"center_relative_to_perch_top_m":z[:3],"position_error_m":errs[0]})
 rows.append({"lane":e,"worst":worst,"selected":details})
(p/"state-transfer-audit.json").write_text(json.dumps({"rows":rows,"scope":"Compare replay-derived physical states to separately reconstructed hip-roll kinematics after translating both crow and stationary perch; does not validate material, warm-start or GPU host binary identity."},indent=2)+"\n")
for row in rows:print(row["lane"],row["worst"],row["selected"])
