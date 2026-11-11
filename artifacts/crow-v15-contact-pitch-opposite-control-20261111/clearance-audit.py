from pathlib import Path
import collections,gzip,json,math
A=Path("artifacts/crow-v15-contact-pitch-opposite-control-20261111/autonomous-20261126")
with gzip.open(A/"replay.json.gz","rt") as f:d=json.load(f)["payload"]
assert d["environment"]==7 and d["body_names"][40]=="crow_course_perch" and d["body_state_stride"]==13
ends=json.loads((A/"run.stdout").read_text())["first_episode_navigation"];end=ends["steps_by_environment"][7]
values=collections.defaultdict(lambda:[0.]*8);coverage=collections.Counter();overflow=[];samples={}
with gzip.open(A/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);
   if x["lane"]==7:samples[x["step"]]=x
  elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=line.split();t,k,e=map(int,z[1:4]);
   if e==7 and t<=end:
    coverage[t,k]+=1
    if int(z[4])>int(z[5]):overflow.append([t,k])
  elif line.startswith("CROW_SCENE_CONTACT "):
   z=line.split();t,k,e,a,b=map(int,z[1:6]);
   if e!=7 or t>end or 40 not in (a,b):continue
   body=b if a==40 else a;sgn=1. if a==40 else -1.;normal,tu,tv=map(float,z[6:9]);nx,ny,nz=map(float,z[13:16]);tx,ty,tz=map(float,z[16:19]);up=sgn*(nz*normal+tz*tu+(nx*ty-ny*tx)*tv)
   side=0 if body==11 or 19<=body<=26 else 1 if body==15 or 27<=body<=34 else -1;v=values[t]
   if side<0:v[7]+=normal;continue
   v[side]+=up;v[2+side]+=normal
   if body>=19:
    digit=((body-19)%8)//2
    if side==0:v[4 if digit==3 else 5]+=sgn*nx*normal
    else:v[6]+=sgn*nx*normal if digit==3 else 0
# Read right front normals independently; keep the representation explicit.
rightfront=collections.defaultdict(float)
with gzip.open(A/"run.stderr.gz","rt") as f:
 for line in f:
  if not line.startswith("CROW_SCENE_CONTACT "):continue
  z=line.split();t,k,e,a,b=map(int,z[1:6]);
  if e!=7 or t>end or 40 not in (a,b):continue
  body=b if a==40 else a
  if 27<=body<=32:rightfront[t]+=(1. if a==40 else -1.)*float(z[6])*float(z[13])
def rot(q):
 x,y,z,w=q
 return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def separation(obj,perch,half):
 P=rot(obj[3:7]);Q=rot(perch[3:7]);p=[list(x) for x in zip(*P)];q=[list(x) for x in zip(*Q)];delta=[a-b for a,b in zip(perch[:3],obj[:3])];axes=p+q+[cross(x,y) for x in p for y in q];out=[]
 for axis in axes:
  n=math.sqrt(dot(axis,axis))
  if n>1e-8:
   u=[v/n for v in axis];out.append(abs(dot(delta,u))-sum(abs(dot(x,u))*h for x,h in zip(p,half))-sum(abs(dot(x,u))*h for x,h in zip(q,[.1,.72,.055])))
 return max(out)
nom=.45006*9.81*.02;records=[];parity=[]
for f in d["frames"]:
 t=f["step"]+1 # accepted endpoint samples are indexed one after replay frames
 if not 3000<=t<=end:continue
 x=samples.get(t);assert x is not None
 parity.append(max(abs(a-b) for a,b in zip(x["q"]+x["v"],f["q"]+f["v"])))
 b=f["body_states"];pose=lambda i:b[i*13:i*13+7];perch=pose(40)
 gap=min([separation(pose(i),perch,[.009,.0025,.0092]) for i in range(19,27)]+[separation(pose(11),perch,[.01,.01725,.0092])]);v=values[t];q=f["q"];u=f["v"];tilt=math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))));speed=math.sqrt(sum(x*x for x in u[:3]));complete=all(coverage[t,k]==1 for k in range(16));right=v[1]>=.5*nom and rightfront[t]>1e-6 and v[6]<-1e-6;bilateral=v[0]>0 and v[1]>0 and v[0]+v[1]>=.5*nom and v[5]>1e-6 and v[4]<-1e-6 and rightfront[t]>1e-6 and v[6]<-1e-6
 stable=complete and v[7]<=1e-7 and tilt<=.35 and speed<=.12
 records.append({"step":t,"gap_m":gap,"right_supported":right,"bilateral_supported":bilateral,"release":stable and right and v[2]<=1e-7 and gap>=.001,"grip":stable and bilateral})
def spans(key):
 result=[];start=None
 for x in records:
  if x[key] and start is None:start=x["step"]
  if not x[key] and start is not None:result.append([start,x["step"]-1]);start=None
 if start is not None:result.append([start,records[-1]["step"]])
 return result
if not parity:
 out={"lane":7,"first_episode_steps":end,"physical_failure":ends["physical_failure_by_environment"][7],"insufficient_late_frames":True,"scope":"No release clearance result: first episode ended before the late analysis window."}
 (A/"clearance-audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2));raise SystemExit(0)
assert max(parity)<1e-6, max(parity)
out={"endpoint_replay_offset":1,"endpoint_q_v_max_error":max(parity),"lane":7,"first_episode_steps":end,"physical_failure":ends["physical_failure_by_environment"][7],"release_spans":spans("release"),"bilateral_grip_spans":spans("grip"),"largest_minimum_left_gap_m":max((x["gap_m"] for x in records),default=None),"last_100_bilateral_grip":len(records)>=100 and all(x["grip"] for x in records[-100:]),"missing_substeps":sum(coverage[t,k]!=1 for t in range(1,end+1) for k in range(16)),"overflow":overflow,"scope":"Only lane 7. Foot/toe clearance is separating-axis clearance at control endpoints for the estimated box shapes; support impulses and no-left-contact are summed over every substep. Right support requires opposing normals. No claim of continuous geometric clearance between endpoints, an actual release attempt, biological realism or unused-layout success."}
(A/"clearance-audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
