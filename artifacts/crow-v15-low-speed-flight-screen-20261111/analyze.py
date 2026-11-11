from pathlib import Path
import collections,gzip,json,math,statistics
D=Path("artifacts/crow-v15-low-speed-flight-screen-20261111");data=json.loads((D/"flight.stdout").read_text());first=data["first_episode_navigation"];ends=first["steps_by_environment"];samples=collections.defaultdict(list);coverage=collections.Counter();toe=collections.Counter();after=collections.Counter();ground=collections.Counter();overflow=[];errors=[]
with gzip.open(D/"flight.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_FLIGHT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
   if 1000<=t<=min(1500,ends[e]):samples[e].append(x)
  elif line.startswith("CROW_TOE_V_SUBSTEP "):
   z=line.split();t,k,e=map(int,z[1:4]);
   if t<=ends[e] and max(abs(float(v)) for v in z[4:])>=11.99:
    toe[e]+=1;after[e]+=int(t>=400)
  elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=line.split();t,k,e=map(int,z[1:4]);
   if t<=ends[e]:
    coverage[e,t,k]+=1
    if int(z[4])>int(z[5]):overflow.append([e,t,k])
    if t>=400 and sum(float(v) for v in z[8:11])>1e-7:ground[e]+=1
  elif line.startswith("CROW_SUBSTEP_ERROR"):errors.append(line.strip())
rows=[]
for e in range(12):
 target=[-.05,0.,.05,.10,.15,.25][e%6];vals=[]
 for x in samples[e]:
  q=x["q"];v=x["v"];a,b,c,w=q[3:7]; yaw=math.atan2(2*(w*c+a*b),1-2*(b*b+c*c));forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1];pitch=math.asin(max(-1.,min(1.,2*(w*b-c*a))));roll=math.atan2(2*(w*a+b*c),1-2*(a*a+b*b));vals.append({"step":x["step"],"forward_m_s":forward,"lateral_m_s":lateral,"height_coordinate_m":q[2],"pitch_rad":pitch,"roll_rad":roll,"yaw_rad":yaw})
 rows.append({"lane":e,"effective_speed_target_m_s":target,"first_episode_steps":ends[e],"physical_failure":first["physical_failure_by_environment"][e],"toe_cap_substeps":toe[e],"toe_cap_after_step_400":after[e],"ground_impulse_substeps_after_400":ground[e],"missing_or_duplicate_substeps":sum(coverage[e,t,k]!=1 for t in range(1,ends[e]+1) for k in range(16)),"sampled_tracking":vals,"mean_sampled_forward_m_s":statistics.mean(x["forward_m_s"] for x in vals) if vals else None,"max_sampled_speed_error_m_s":max((abs(x["forward_m_s"]-target) for x in vals),default=None)})
out={"rows":rows,"failed_environment_steps":data["failed_environment_steps"],"contact_overflows":overflow,"substep_errors":errors,"limitations":"Five-second windows are sampled at 100-control-step spacing, not continuous flight tracking; height is root-coordinate, not calibrated physical altitude. The effective target follows the algebra of the journey adapter and has not been independently instrumented. Course collisions disabled; fixed straight diagnostic commands replace navigation; two starts per target on a used seed do not qualify perturbation recovery, turns, landing or biological flight."};(D/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
for r in rows:print(r["lane"],"target",r["effective_speed_target_m_s"],"end",r["first_episode_steps"],"fail",r["physical_failure"],"samples",len(r["sampled_tracking"]),"mean_v",r["mean_sampled_forward_m_s"],"max_error",r["max_sampled_speed_error_m_s"],"toe caps",r["toe_cap_substeps"],"ground after400",r["ground_impulse_substeps_after_400"],"missing",r["missing_or_duplicate_substeps"])
print("failed steps",data["failed_environment_steps"],"overflows",len(overflow),"substep errors",errors[-3:])
