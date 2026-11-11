from pathlib import Path
import gzip,json,math,os,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-landing-arrival-com-comparison-20261111"
trials=[("failed",R/"artifacts/crow-v15-matched-release-inhibited-20261111/autonomous-20261126/replay.json.gz",[3200,3240,3280,3320,3360,3400,3440,3480,3508]),("prior-supported",R/"artifacts/crow-v15-aligned-contact-release-control-20261111/autonomous-20261126/replay.json.gz",[3200,3240,3280,3320,3360,3400,3440,3480,3508,3600,3700,3800])]
rows=[];inputs=[]
for e,(name,path,steps) in enumerate(trials):
 with gzip.open(path,"rt") as f:d=json.load(f)["payload"]
 assert d["environment"]==7 and d["body_names"][40]=="crow_course_perch" and d["body_state_stride"]==13
 for t in steps:
  x=d["frames"][t-1];b=x["body_states"];pose=lambda i:b[i*13:i*13+3]
  q=x["q"];v=x["v"];a,y,c,w=q[3:7]
  rows.append({"trial":name,"step":t,"root_x":q[0],"pitch_rad":math.asin(max(-1,min(1,2*(w*y-c*a)))),"pitch_rate_rad_s":v[4],"left_foot_x":pose(11)[0],"right_foot_x":pose(15)[0],"perch_x":pose(40)[0]})
  inputs.append(str(e)+" "+str(t)+" "+" ".join(map(str,q+v)))
env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")}
r=subprocess.run([str(R/"build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics")],input="\n".join(inputs)+"\n",text=True,capture_output=True,env=env,check=True)
com={}
for line in r.stdout.splitlines():
 z=line.split()
 if z[0]=="com":com[(int(z[1]),int(z[2]))]=[float(v) for v in z[3:]]
for x in rows:
 e=0 if x["trial"]=="failed" else 1
 x["total_mass_kg"],x["com_x"],x["com_y"],x["com_z"]=com[e,x["step"]]
 x["com_minus_feet_midpoint_x_m"]=x["com_x"]-.5*(x["left_foot_x"]+x["right_foot_x"])
out={"rows":rows,"scope":"Independent articulated-body COM from sampled replay q/v, two used development trials, lane 7. Foot body centers are not measured centers of pressure; this comparison does not establish causality, full contact support, biological validity or unseen-layout success."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
for x in rows:print(x["trial"],x["step"],"pitch",round(x["pitch_rad"],3),"COM minus feet",round(x["com_minus_feet_midpoint_x_m"],3),"COM minus perch",round(x["com_x"]-x["perch_x"],3))
