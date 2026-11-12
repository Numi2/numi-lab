from pathlib import Path
import gzip,json,math
p=Path("artifacts/crow-v15-preimpact-immediate-toe-contact-20261112/high");states={};contacts=[];controls={};failure=[]
with gzip.open(p/"trace.log.gz","rt") as f:
 for l in f:
  z=l.split()
  if not z:continue
  if z[0]=="step" and z[1]=="1":
   t=int(z[2]);a=z.index("q");b=z.index("v");states[t]={"q":list(map(float,z[a+1:b])),"v":list(map(float,z[b+1:]))}
  elif z[0]=="control" and z[1]=="1":controls[int(z[2])]=list(map(float,z[3:]))
  elif z[0]=="contact" and z[1]=="1" and float(z[5])>0:contacts.append({"step":int(z[2]),"body_a":z[3],"body_b":z[4],"normal_impulse_ns":float(z[5]),"separation_m":float(z[6])})
with gzip.open(p/"substeps.log.gz","rt") as f:
 for l in f:
  if l.startswith(("failure","CROW_SUBSTEP_ERROR")):failure.append(l.strip())
last=max(states);x=states[last];q=x["q"];v=x["v"];roll=math.atan2(2*(q[6]*q[3]+q[4]*q[5]),1-2*(q[3]**2+q[4]**2));pitch=math.asin(max(-1,min(1,2*(q[6]*q[4]-q[5]*q[3]))));yaw=math.atan2(2*(q[6]*q[5]+q[3]*q[4]),1-2*(q[4]**2+q[5]**2));first=min(c["step"] for c in contacts) if contacts else None
out={"failing_lane":1,"last_accepted_step":last,"last_root_position_perch_frame_m":q[:3],"last_root_velocity_m_s":v[:3],"last_root_angular_velocity_rad_s":v[3:6],"last_attitude_roll_pitch_yaw_rad":[roll,pitch,yaw],"last_recorded_toe_controller_values":controls.get(last),"first_accepted_contact_step":first,"first_accepted_named_contacts":[x for x in contacts if x["step"]==first],"failure_messages":failure,"limitations":"The native failure output does not identify the failing collider pair or exact ordering within the failed substep. The recorded control values are toe target/latch data, not full applied actuator efforts. The transferred run does not continue the original controller or warm-start state."}
(p.parent/"failure-diagnostic.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
