from pathlib import Path
import collections,hashlib,json,math,subprocess
A=Path("artifacts/crow-v15-moderate-full-journey-dev-20261112");D=A/"candidate-20261127-rebound-flexed";manifest=json.loads((D/"stderr-parts.json").read_text());hs=hashlib.sha256();files=[]
for p in manifest["parts"]:
 f=D/p["name"]; datahash=hashlib.sha256();n=0
 with f.open("rb") as b:
  while True:
   chunk=b.read(1024*1024)
   if not chunk:break
   hs.update(chunk);datahash.update(chunk);n+=len(chunk)
 assert n==p["bytes"] and datahash.hexdigest()==p["sha256"],p
 files.append(str(f))
assert hs.hexdigest()==manifest["compressed_sha256"]
d=json.loads((D/"run.stdout").read_text());first=d["first_episode_navigation"];ends=first["steps_by_environment"]
sums=collections.defaultdict(lambda:[0.]*8);coverage=collections.Counter();caps=collections.Counter();errors=[];overflow=[];previous={};last={};hit={}
def brief(x):
 q=x["q"];v=x["v"];a,b,c,w=q[3:7]
 return {"step":x["step"],"position_m":q[:3],"attitude_roll_pitch_yaw_rad":[math.atan2(2*(w*a+b*c),1-2*(a*a+b*b)),math.asin(max(-1.,min(1.,2*(w*b-c*a)))),math.atan2(2*(w*c+a*b),1-2*(b*b+c*c))],"linear_velocity_m_s":v[:3],"angular_velocity_rad_s":v[3:6],"policy_action":x["policy_action"]}
cat=subprocess.Popen(["cat",*files],stdout=subprocess.PIPE);z=subprocess.Popen(["gzip","-dc"],stdin=cat.stdout,stdout=subprocess.PIPE,text=True);cat.stdout.close()
for line in z.stdout:
 if line.startswith("CROW_SUPPORT_SAMPLE "):
  x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"]
  if t>ends[e]+1:continue
  if t>=356 and e not in hit:
   hits=[c for c in x["contacts"] if c["impulses"][0]>1e-7 and "locomotion_ground" not in (c["a_name"],c["b_name"])]
   if hits:hit[e]={"step":t,"pairs":[[c["a_name"],c["b_name"]] for c in hits],"impulses":[c["impulses"] for c in hits],"state":brief(x),"previous_state":previous.get(e)}
  previous[e]=brief(x)
  if t==ends[e]:last[e]=brief(x)
 elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
  v=line.split();t,k,e=map(int,v[1:4])
  if t<=ends[e]:
   coverage[e,t,k]+=1
   if int(v[4])>int(v[5]):overflow.append([e,t,k,int(v[4]),int(v[5])])
 elif line.startswith("CROW_TOE_V_SUBSTEP "):
  v=line.split();t,k,e=map(int,v[1:4])
  if t<=ends[e] and max(abs(float(x)) for x in v[4:])>=11.99:caps[e]+=1
 elif line.startswith("CROW_SCENE_CONTACT "):
  v=line.split();t,k,e,ba,bb=map(int,v[1:6])
  if t>ends[e] or 40 not in (ba,bb):continue
  body=bb if ba==40 else ba;sign=1. if ba==40 else -1.;normal,tan1,tan2=map(float,v[6:9]);nx,ny,nz=map(float,v[13:16]);tx,ty,tz=map(float,v[16:19]);vz=nx*ty-ny*tx;upward=sign*(nz*normal+tz*tan1+vz*tan2)
  side=0 if body==11 or 19<=body<=26 else 1 if body==15 or 27<=body<=34 else -1;o=sums[e,t]
  if side<0:o[6]+=normal;continue
  o[side]+=upward
  if body>=19:
   digit=((body-19)%8)//2;o[2+2*side+(1 if digit==3 else 0)]+=sign*nx*normal
  o[7]+=normal
 elif "CROW_SUBSTEP_ERROR" in line:errors.append(line.strip())
z.stdout.close();assert z.wait()==0 and cat.wait()==0
nom=.45006*9.81*.02
def longest(seq):
 best=run=0
 for v in seq:run=run+1 if v else 0;best=max(best,run)
 return best
rows=[]
for e,end in enumerate(ends):
 vals=[sums[e,t] for t in range(1,end+1)];support=[x[0]+x[1]>=.5*nom and x[6]<=1e-7 for x in vals];grip=[ok and x[2]>1e-6 and x[3]<-1e-6 and x[4]>1e-6 and x[5]<-1e-6 for ok,x in zip(support,vals)]
 rows.append({"lane":e,"first_episode_steps":end,"physical_failure":first["physical_failure_by_environment"][e],"waypoints":first["maximum_waypoints_by_environment"][e],"first_observed_non_ground_contact":hit.get(e),"last_observed_state":last.get(e),"first_perch_foot_impulse_step":next((i+1 for i,x in enumerate(vals) if x[7]>1e-7),None),"max_consecutive_half_weight_nonbody_intervals":longest(support),"max_consecutive_half_weight_bilateral_opposed_intervals":longest(grip),"last_25_half_weight_bilateral_opposed":len(grip)>=25 and all(grip[-25:]),"nonfoot_perch_impulse_intervals":sum(x[6]>1e-7 for x in vals),"missing_or_duplicate_substeps":sum(coverage[e,t,k]!=1 for t in range(1,end+1) for k in range(16)),"toe_speed_limit_substeps":caps[e]})
out={"teacher":d["birdflow_journey_teacher"],"action_source":d["action_source"],"host_requested_resets":d["host_requested_resets"],"first_episode_navigation":first,"rows":rows,"substep_errors":errors[-10:],"contact_capacity_overflows":overflow,"limitations":"Development seed/layout. Contact identities use endpoint observations; first collision ordering and missed substep-only contacts are not established. Actor outputs are not applied actuator targets. Opposed contact normals do not alone establish biological grip. Perch indices assume the recorded course body layout. Coverage, overflow, clearance and final stability require independent checks."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
