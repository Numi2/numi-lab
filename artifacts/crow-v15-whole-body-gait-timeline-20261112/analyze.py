from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-whole-body-gait-timeline-20261112")
P=Path("artifacts/crow-v15-gait-target-observer-20261112/autonomous-20261126")
r=json.load(gzip.open(P/"replay.json.gz","rt"))["payload"]
assert r["environment"]==5 and r["timestep_seconds"]==.02
states={int(f["step"])+1:f for f in r["frames"] if 201<=int(f["step"])+1<=300}
assert len(states)==100
contacts={};qsub={}
with gzip.open(P/"run.stderr.gz","rt") as f:
 for line in f:
  z=line.split()
  if not z or z[0] not in ("CROW_TWO_TOE_SUBSTEP","CROW_Q_SUBSTEP"): continue
  t,k,e=map(int,z[1:4])
  if e!=5 or not 201<=t<=300:continue
  if z[0]=="CROW_TWO_TOE_SUBSTEP":
   assert (t,k) not in contacts and int(z[4])<=int(z[5]);contacts[t,k]=list(map(float,z[6:]));assert len(contacts[t,k])==37
  elif k==15:
   assert t not in qsub;qsub[t]=list(map(float,z[4:]));assert len(qsub[t])==41
assert len(contacts)==1600 and len(qsub)==100
error=max(abs(a-b) for t,f in states.items() for a,b in zip(f["q"],qsub[t]));assert error<1e-6,error
names=r["body_names"];stride=r["body_state_stride"]
def body(f,name):
 i=names.index(name)*stride;return f["body_states"][i:i+stride]
def euler(q):
 x,y,z,w=q
 return [math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))]
rows=[]
for t in range(201,301):
 f=states[t];tr=body(f,"crow_body");hd=body(f,"crow_head");lf=body(f,"crow_left_foot");rf=body(f,"crow_right_foot");c=[contacts[t,k] for k in range(16)];q=f["q"]
 rows.append({"contact_interval":t,"replay_step_label":f["step"],"contact_interval_end_time_s":t*.02,"trunk_position_m":tr[:3],"trunk_roll_pitch_yaw_rad":euler(tr[3:7]),"head_position_m":hd[:3],"head_minus_trunk_world_m":[hd[i]-tr[i] for i in range(3)],"head_roll_pitch_yaw_rad":euler(hd[3:7]),"left_foot_position_m":lf[:3],"right_foot_position_m":rf[:3],"left_hip_knee_ankle_rad":q[14:17],"right_hip_knee_ankle_rad":q[17:20],"left_ground_impulse_Ns":sum(x[2] for x in c),"right_ground_impulse_Ns":sum(x[3] for x in c),"unsupported_substeps":sum(x[2]<=1e-7 and x[3]<=1e-7 for x in c),"nonfoot_ground_impulse_Ns":sum(x[4] for x in c)})
def extent(vals):return max(vals)-min(vals)
summary={"lane":5,"sampled_intervals":len(rows),"contact_substeps":len(contacts),"max_q_substep_endpoint_error":error,"root_forward_displacement_m":rows[-1]["trunk_position_m"][0]-rows[0]["trunk_position_m"][0],"trunk_vertical_range_m":extent([x["trunk_position_m"][2] for x in rows]),"trunk_roll_range_rad":extent([x["trunk_roll_pitch_yaw_rad"][0] for x in rows]),"trunk_pitch_range_rad":extent([x["trunk_roll_pitch_yaw_rad"][1] for x in rows]),"head_world_x_range_m":extent([x["head_position_m"][0] for x in rows]),"head_minus_trunk_world_x_range_m":extent([x["head_minus_trunk_world_m"][0] for x in rows]),"left_contact_impulse_Ns":sum(x["left_ground_impulse_Ns"] for x in rows),"right_contact_impulse_Ns":sum(x["right_ground_impulse_Ns"] for x in rows),"unsupported_substeps":sum(x["unsupported_substeps"] for x in rows),"nonfoot_ground_impulse_Ns":sum(x["nonfoot_ground_impulse_Ns"] for x in rows)}
out={"summary":summary,"timeline":rows,"limits":"Replay frame labels are one less than contact-interval labels; matching final-substep q confirms this join. One retained development lane, sampled endpoint poses and interval impulses; root is not COM. No biological target or gait qualification."}
(A/"timeline.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(summary,indent=2))
