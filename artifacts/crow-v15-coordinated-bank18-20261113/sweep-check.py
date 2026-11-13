from pathlib import Path
import gzip,json,sys
P=Path("artifacts/crow-v15-coordinated-bank18-20261113")/sys.argv[1];meta=json.loads((P/"run.stdout").read_text())["first_episode_navigation"];samples={};states={};velocities={}
def smooth(a,b,x):
 u=max(0.,min(1.,(x-a)/(b-a)));return u*u*(3.-2.*u)
with gzip.open(P/"run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(l.split(" ",1)[1]);states[x["lane"],x["step"]]=x
  if l.startswith("CROW_BODY_JOINT_V_SUBSTEP "):
   z=l.split();t,k,e=map(int,z[1:4]);assert (e,t,k) not in velocities;velocities[e,t,k]=list(map(float,z[4:]))
  if l.startswith("CROW_DRIVE_TARGET "):
   z=l.split();t,e=map(int,z[1:3]);assert (e,t) not in samples;samples[e,t]=list(map(float,z[3:]))
rows=[]
for e in (6,7):
 end=min(meta["steps_by_environment"][e],1800);errs=[]
 for t in range(435,end+1):
  sec=(t-2)*.02;v=samples.get((e,t));expected=(1 if e==6 else -1)*.10*smooth(12,12.5,sec)*(1-smooth(23.5,24,sec))
  if v is not None:errs.append(abs((v[6]-v[7])/2-expected))
 positions=[abs(states[e,t]["q"][j]) for t in range(500,end+1) if (e,t) in states for j in (7,8)]
 speeds=[abs(velocities[e,t,k][j]) for t in range(500,end+1) for k in range(16) if (e,t,k) in velocities for j in (0,1)]
 target_values=[abs(samples[e,t][j]) for t in range(435,end+1) if (e,t) in samples for j in (6,7)]
 good=len(errs)==max(0,end-434) and bool(errs) and max(errs)<.0003 and len(positions)==2*max(0,end-499) and max(positions,default=1)<.2001 and len(speeds)==32*max(0,end-499) and max(speeds,default=100)<14.999 and len(target_values)==2*max(0,end-434) and max(target_values,default=1)<.2001
 rows.append({"lane":e,"expected_count":max(0,end-434),"samples":len(errs),"maximum_error_rad":max(errs,default=None),"max_accepted_position_abs_rad":max(positions,default=None),"max_substep_speed_abs_rad_s":max(speeds,default=None),"max_target_abs_rad":max(target_values,default=None),"passed":good})
out={"rows":rows,"passed":all(x["passed"] for x in rows)};(P/"sweep-gate.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
