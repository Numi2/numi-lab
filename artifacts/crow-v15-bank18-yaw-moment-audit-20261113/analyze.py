from pathlib import Path
import gzip,json,os,subprocess,hashlib,math,statistics
R=Path.cwd();D=R/"artifacts/crow-v15-bank18-yaw-moment-audit-20261113";source=R/"artifacts/crow-v15-coordinated-bank18-20261113/signed-turn-20261126-rebound-bank18/run.stderr.gz";exe=R/"build/crow-v15-bank-yaw-moment-audit-20261113/kinematics";lib=R/"build/crow-v15-mirrored-leg-isolation-20261113";kin={};wrench={};total={}
with gzip.open(source,"rt") as f:
 for line in f:
  z=line.split()
  if not z or z[0] not in ("CROW_AERO_INPUT","CROW_KIN_STATE","CROW_AERO_WRENCH","CROW_AERO_FORCE"):continue
  t,k,e=map(int,z[1:4]);key=(e,t,k)
  if e not in (6,7) or not 650<=t<=1150:continue
  if z[0]=="CROW_AERO_INPUT":
   assert key not in kin;v=list(map(float,z[4:]));assert len(v)==81 and all(map(math.isfinite,v));kin[key]=v
  elif z[0]=="CROW_KIN_STATE":
   assert z[4:6]==["41","40"];v=list(map(float,z[6:]));assert key in kin and len(v)==81 and max(abs(a-b) for a,b in zip(kin[key],v))<1e-7
  elif z[0]=="CROW_AERO_WRENCH":
   body=int(z[4]);assert body in (0,5,6,7) and key+(body,) not in wrench;v=list(map(float,z[5:]));assert len(v)==6 and all(map(math.isfinite,v));wrench[key+(body,)]=v
  else:
   v=list(map(float,z[5:]));assert len(v)==12 and v[-1]==4;total[key]=v
keys=sorted(kin);assert len(keys)==2*501*16 and len(total)==len(keys) and len(wrench)==len(keys)*4
inputs="\n".join(str(i)+" 0 "+" ".join(map(str,kin[k])) for i,k in enumerate(keys))+"\n";r=subprocess.run([str(exe)],input=inputs,text=True,capture_output=True,check=True,env={**os.environ,"DYLD_LIBRARY_PATH":str(lib)});rows={6:{"yaw":[],"arm_yaw":[],"com_torque_yaw":[],"force_residual":[]},7:{"yaw":[],"arm_yaw":[],"com_torque_yaw":[],"force_residual":[]}};count=0
for line in r.stdout.splitlines():
 z=line.split();i=int(z[1]);v=list(map(float,z[3:]));assert len(v)==19;key=keys[i];com=v[1:4];positions={body:v[7+3*j:10+3*j] for j,body in enumerate((0,5,6,7))};arm=0.;torque=0.;net=[0.,0.,0.]
 for body in (0,5,6,7):
  w=wrench[key+(body,)];p=positions[body];arm+=(p[0]-com[0])*w[1]-(p[1]-com[1])*w[0];torque+=w[5]
  for j in range(3):net[j]+=w[j]
 q=rows[key[0]];q["yaw"].append(arm+torque);q["arm_yaw"].append(arm);q["com_torque_yaw"].append(torque);q["force_residual"].append(max(abs(net[j]-total[key][j]) for j in range(3)));count+=1
assert count==len(keys)
def summary(x):return {"mean":statistics.mean(x),"minimum":min(x),"maximum":max(x),"rms":math.sqrt(statistics.mean(y*y for y in x))}
out=[]
for e,q in rows.items():
 assert len(q["yaw"])==501*16 and max(q["force_residual"])<1e-7;out.append({"lane":e,"samples":len(q["yaw"]),"total_aero_yaw_moment_nm":summary(q["yaw"]),"body_com_torque_sum_nm":summary(q["com_torque_yaw"]),"moment_arm_sum_nm":summary(q["arm_yaw"]),"maximum_force_residual_n":max(q["force_residual"])})
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();data={"rows":out,"source_sha256":h(source),"kinematics_sha256":h(exe),"sample_ticks":[650,1150],"substeps_per_control":16,"scope":"Posthoc modeled aerodynamic moment about instantaneous whole-body COM, not total angular momentum or proof of heading cause. Includes moment arms of all four observed aero bodies and their body-COM torques. One used seed and correlated lanes; not biological validation."};(D/"analysis.json").write_text(json.dumps(data,indent=2)+"\n");print(json.dumps(data,indent=2))
