from pathlib import Path
import collections,gzip,json,math
p=Path("artifacts/crow-v15-early-wing-sequence-screen-20261112/autonomous-20261126")
out=Path("artifacts/crow-v15-early-wing-sequence-screen-20261112")
d=json.loads((p/"run.stdout").read_text())["first_episode_navigation"]; ends=d["steps_by_environment"]
phase=collections.defaultdict(dict); snapshots=collections.defaultdict(dict); impulses=collections.defaultdict(lambda:[0.]*8); impacts=collections.defaultdict(list);coverage=collections.Counter();errors=[];overflow=[]
with gzip.open(p/"run.stderr.gz","rt") as f:
 for line in f:
  tag=line.split(" ",1)[0]
  if tag=="CROW_PHASE_SAMPLE":
   a=line.split();t,e,status=int(a[1]),int(a[2]),int(a[3])
   if t<=ends[e]:phase[e][t]=[status]+list(map(float,a[4:]))
  elif tag=="CROW_SUPPORT_SAMPLE":
   x=json.loads(line.split(" ",1)[1]);t,e=x["step"],x["lane"]
   if t<=ends[e] and t>=3000:snapshots[e][t]={k:x[k] for k in ("q","v","policy_action")}
  elif tag=="CROW_TWO_TOE_SUBSTEP":
   z=line.split();t,k,e=map(int,z[1:4])
   if t<=ends[e]:
    coverage[e,t,k]+=1
    if int(z[4])>int(z[5]):overflow.append([e,t,k,int(z[4]),int(z[5])])
  elif tag=="CROW_SCENE_CONTACT":
   z=line.split();t,k,e,ba,bb=map(int,z[1:6])
   if t>ends[e]:continue
   normal,tan1,tan2=map(float,z[6:9])
   bird=ba if ba<35 else bb
   if normal>1e-7 and bird not in (11,15) and not 19<=bird<=34: impacts[e].append([t,k,ba,bb,normal])
   if 40 not in (ba,bb):continue
   body=bb if ba==40 else ba;sign=1. if ba==40 else -1.
   nx,ny,nz=map(float,z[13:16]);tx,ty,tz=map(float,z[16:19]);vz=nx*ty-ny*tx
   upward=sign*(nz*normal+tz*tan1+vz*tan2)
   if body==11 or 19<=body<=26:side=0
   elif body==15 or 27<=body<=34:side=1
   else:side=-1
   o=impulses[e,t]
   if side<0:o[6]+=normal;continue
   o[side]+=upward
   if body>=19:
    digit=((body-19)%8)//2;o[2+2*side+(1 if digit==3 else 0)]+=sign*nx*normal
   o[7]+=normal
  elif "CROW_SUBSTEP_ERROR" in line:errors.append(line.strip())
def bouts(vals):
 result=[];start=None
 for t,flag in vals:
  if flag and start is None:start=t
  if not flag and start is not None:result.append([start,t-1]);start=None
 if start is not None:result.append([start,vals[-1][0]])
 return result
rows=[];nom=.45006*9.81*.02
for e,end in enumerate(ends):
 q=phase[e]; transitions=[];last=None
 for t,v in sorted(q.items()):
  ph=v[1]
  if ph!=last:transitions.append([t,ph]);last=ph
 vals=[(t,impulses[e,t]) for t in range(1,end+1)]
 support=[(t,(x[0]+x[1]>=.5*nom and x[6]<=1e-7)) for t,x in vals]
 grip=[(t,ok and x[2]>1e-6 and x[3]<-1e-6 and x[4]>1e-6 and x[5]<-1e-6) for (t,ok),(_,x) in zip(support,vals)]
 gb=bouts(grip);firstfoot=next((t for t,x in vals if x[7]>1e-7),None)
 # Preserve event windows rather than interpreting the unlabeled raw state components as forces.
 events=sorted(set(t for t in [firstfoot,*(a for a,b in gb if b-a+1>=10),end-10,end] if t is not None and t>=1))
 event_data=[]
 for t in events:
  nearby=next((u for u in (t,t-1,t+1) if u in snapshots[e]),None)
  s=snapshots[e].get(nearby,{})
  event_data.append({"step":t,"phase_raw":q.get(t),"support_impulse":impulses[e,t],"state_step":nearby,"root_q":s.get("q",[])[:7],"root_v":s.get("v",[])[:6],"action":s.get("policy_action")})
 rows.append({"lane":e,"end":end,"terminated":d["physical_failure_by_environment"][e],"reason":d["termination_reason_by_environment"][e],"phase_transitions":transitions,"first_foot_contact":firstfoot,"support_bouts_10plus":[b for b in bouts(support) if b[1]-b[0]+1>=10],"grip_bouts_10plus":[b for b in gb if b[1]-b[0]+1>=10],"events":event_data,"last_50_nonfoot_collisions": [i for i in impacts[e] if i[0]>=end-50][:30],"missing_substeps":sum(coverage[e,t,k]!=1 for t in range(1,end+1) for k in range(16)),"phase_count":len(q)})
r={"trace":str(p/"run.stderr.gz"),"rows":rows,"errors":errors,"overflows":overflow,"phase_fields":["status.w","threatTeacher.w","nutrition.x","nutrition.y","commandAndPhase.y","commandAndPhase.z","threatGeometry.y","threatGeometry.z","threatGeometry.w"],"limits":"development episodes; contact sampled at substep boundaries; phase samples and q/v snapshots have not been shown to be temporally identical within an interval"}
(out/"analysis.json").write_text(json.dumps(r,indent=2)+"\n")
for row in rows:
 print(json.dumps({k:v for k,v in row.items() if k not in ("events","last_50_nonfoot_collisions")}))
 print("events",json.dumps([{k:v for k,v in x.items() if k not in ("action","support_impulse")} for x in row["events"]]))
 print("impacts",json.dumps(row["last_50_nonfoot_collisions"][:8]))
print("errors",len(errors),"overflow",len(overflow))
