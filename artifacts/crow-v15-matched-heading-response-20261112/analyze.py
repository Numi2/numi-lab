from pathlib import Path
import collections,gzip,hashlib,json,math
A=Path("artifacts/crow-v15-matched-heading-response-20261112");B=Path("artifacts/crow-v15-continuous-launch-open-diagnostic-20261112");name="candidate-20261126-rebound-hip"
mp=Path("artifacts/crow-v15-drive-work-observer-20261112/dof-map.txt");limits={}
for l in mp.read_text().splitlines():
 z=l.split();limits[int(z[0])]=(z[1],float(z[-1]))
def read(p):
 meta=json.loads((p/name/"run.stdout").read_text())["first_episode_navigation"];ends=meta["steps_by_environment"];state={};contacts={};speeds=collections.Counter();nonfoot=collections.Counter();errors=[]
 with gzip.open(p/name/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"]
    if t<=ends[e]+1:state[e,t]=(x,line)
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);
    if t<=ends[e]+1:
     key=e,t;h=contacts.setdefault(key,hashlib.sha256());h.update(line.encode());nonfoot[e]+=int(t>=601 and float(z[10])>1e-7)
   elif line.startswith("CROW_BODY_JOINT_V_SUBSTEP ") or line.startswith("CROW_TOE_V_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);inds=(list(range(6,22))+[38,39]) if z[0]=="CROW_BODY_JOINT_V_SUBSTEP" else list(range(22,38))
    if 601<=t<=ends[e]+1:
     for j,v in zip(inds,map(float,z[4:])):
      if abs(v)>=limits[j][1]-1e-4:speeds[e,limits[j][0]]+=1
   elif line.startswith("CROW_SUBSTEP_ERROR"):errors.append(line.strip())
 return meta,state,{k:v.hexdigest() for k,v in contacts.items()},speeds,nonfoot,errors
a,as_,ac,ah,an,ae=read(A);b,bs,bc,bh,bn,be=read(B)
def pose(x):
 q=x["q"];xx,yy,zz,w=q[3:7];return {"position":q[:3],"velocity":x["v"][:6],"roll":math.atan2(2*(w*xx+yy*zz),1-2*(xx*xx+yy*yy)),"pitch":math.asin(max(-1,min(1,2*(w*yy-zz*xx)))),"yaw":math.atan2(2*(w*zz+xx*yy),1-2*(yy*yy+zz*zz))}
rows=[]
for e in range(8):
 first_state=next((t for t in range(1,1052) if (e,t) not in as_ or (e,t) not in bs or as_[e,t][1]!=bs[e,t][1]),None)
 first_contact=next((t for t in range(201,1052) if ac.get((e,t))!=bc.get((e,t))),None)
 samples=[]
 for t in (600,650,750,850,900,950,1050):
  if (e,t) not in as_ or (e,t) not in bs:continue
  x=pose(as_[e,t][0]);y=pose(bs[e,t][0]);samples.append({"trace_submission":t,"candidate":x,"matched_control":y,"position_difference_m":[x["position"][j]-y["position"][j] for j in range(3)],"roll_difference_rad":x["roll"]-y["roll"],"yaw_difference_rad":math.atan2(math.sin(x["yaw"]-y["yaw"]),math.cos(x["yaw"]-y["yaw"]))})
 rows.append({"lane":e,"first_state_difference":first_state,"first_ground_contact_difference":first_contact,"candidate_episode_steps":a["steps_by_environment"][e],"candidate_reason":a["termination_reason_by_environment"][e],"candidate_nonfoot_ground_substeps_after_12s":an[e],"candidate_speed_cap_substeps_after_12s":{j:n for (i,j),n in ah.items() if i==e},"samples":samples})
out={"rows":rows,"candidate_substep_errors":ae,"control_substep_errors":be,"time_note":"Trace submissions include the one unrecorded policy warmup transition.","scope":"Paired used-seed open-flight pulse; not a heading controller, held-out result, or biological qualification."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
for r in rows:
 z=r["samples"][-1] if r["samples"] else {};print(r["lane"],"first",r["first_state_difference"],r["first_ground_contact_difference"],"reason",r["candidate_reason"],"caps",r["candidate_speed_cap_substeps_after_12s"],"nonfoot",r["candidate_nonfoot_ground_substeps_after_12s"],"last_position_difference",z.get("position_difference_m"),"last_roll_difference",z.get("roll_difference_rad"),"last_yaw_difference",z.get("yaw_difference_rad"))
