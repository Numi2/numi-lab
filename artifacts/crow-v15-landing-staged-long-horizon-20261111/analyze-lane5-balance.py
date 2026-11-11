from pathlib import Path
import gzip,json,math,os,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-landing-staged-long-horizon-20261111/autonomous-20261126";d=json.loads((A/"run.stdout").read_text());end=d["first_episode_navigation"]["steps_by_environment"][5]
with gzip.open(A/"replay.json.gz","rt") as f:r=json.load(f)["payload"]
assert r["environment"]==5 and r["body_names"][40]=="crow_course_perch" and r["body_state_stride"]==13
frames=[x for x in r["frames"] if 3500<=x["step"]<end and x["step"]%25==0];lines=[];rows=[]
for x in frames:
 t=x["step"];q=x["q"];v=x["v"];b=x["body_states"];a,y,c,w=q[3:7];pose=lambda i:b[i*13:i*13+3]
 rows.append({"replay_step":t,"root_x":q[0],"pitch_rad":math.asin(max(-1,min(1,2*(w*y-c*a)))),"pitch_rate_rad_s":v[4],"left_foot_x":pose(11)[0],"right_foot_x":pose(15)[0],"perch_x":pose(40)[0]});lines.append("5 "+str(t)+" "+" ".join(map(str,q+v)))
env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")};p=subprocess.run([str(R/"build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics")],input="\n".join(lines)+"\n",text=True,capture_output=True,env=env,check=True);com={}
for line in p.stdout.splitlines():
 z=line.split()
 if z[0]=="com":com[int(z[2])]=[float(v) for v in z[3:]]
for x in rows:
 x["total_mass_kg"],x["com_x"],x["com_y"],x["com_z"]=com[x["replay_step"]];x["com_minus_perch_x_m"]=x["com_x"]-x["perch_x"];x["com_minus_feet_midpoint_x_m"]=x["com_x"]-.5*(x["left_foot_x"]+x["right_foot_x"])
out={"lane":5,"first_episode_end":end,"samples":rows,"scope":"Estimated articulated-body center of mass from the stored replay, lane 5 on the development course. Foot centers are not centers of pressure; endpoint samples do not establish causal torque, continuous support, biological validity or performance on reserved layouts. Replay step and observer step may differ by one and must not be silently equated."};(A/"lane5-balance.json").write_text(json.dumps(out,indent=2)+"\n")
for x in rows:print(x["replay_step"],round(x["pitch_rad"],3),round(x["pitch_rate_rad_s"],3),round(x["com_minus_perch_x_m"],3),round(x["com_minus_feet_midpoint_x_m"],3))
