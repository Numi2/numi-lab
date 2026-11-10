from pathlib import Path
import gzip,hashlib,json,math,os,statistics,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-flight-reverse-transition-20261110";B=R/"build/crow-v15-flight-reverse-transition-20261110";D=R/"build/crow-v15-toe-damping-isolation-20261110/zero";base=json.loads((R/"artifacts/crow-v15-toe-free-flight-20261110/smoke-command.json").read_text())
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for arm,speed in (("transition",-.15),):
 cmd=list(base)
 for k,v in [("--metallib",str(B/(arm+".metallib"))),("--envs","40"),("--steps","1600"),("--seed","18647329"),("--state-trace",str(B/(arm+".tsv"))),("--state-trace-environment","1"),("--policy-pack",str(R/"build/crow-v15-toe-damping-isolation-20261110/zero-4/zero.policypack"))]:cmd[cmd.index(k)+1]=v
 env={**os.environ,"DYLD_LIBRARY_PATH":str(D),"CROW_CONTACT_TRACE":"1","PYTHONDONTWRITEBYTECODE":"1"};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=900);(A/(arm+".stdout")).write_text(p.stdout)
 with (A/(arm+".stderr.gz")).open("wb") as f:
  with gzip.GzipFile(fileobj=f,mode="wb",mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(str(p.returncode)+" "+p.stderr[-3000:])
 data=json.loads(p.stdout);first=data["first_episode_navigation"];samples=[json.loads(x.split(" ",1)[1]) for x in p.stderr.splitlines() if x.startswith("CROW_FLIGHT_SAMPLE ")];rows=[]
 for lane in range(40):
  mode=lane%5;rate=[0,.03,-.03,0,0][mode];target=[.85,.85,.85,1.05,.65][mode];ss=sorted((x for x in samples if x["lane"]==lane and 1000<=x["step"]<=1500 and x["step"]<=first["steps_by_environment"][lane]),key=lambda x:x["step"]);vals=[]
  for x in ss:
   q=x["q"];v=x["v"];a,b,c,d=q[3:7];yaw=math.atan2(2*(d*c+a*b),1-2*(b*b+c*c));desired=rate*(x["step"]*.02-6);heading=abs(math.atan2(math.sin(desired-yaw),math.cos(desired-yaw)));forward=math.cos(yaw)*v[0]+math.sin(yaw)*v[1];lateral=-math.sin(yaw)*v[0]+math.cos(yaw)*v[1];vals.append((heading,abs(lateral),abs(forward-speed),abs(q[2]-target),q[2]))
  mx=[max((x[i] for x in vals),default=None) for i in range(4)];toe=max((abs(z) for x in samples if x["lane"]==lane and x["step"]<=first["steps_by_environment"][lane] for z in x["v"][22:30]),default=None);passed=(data["failed_environment_steps"]==0 and not first["physical_failure_by_environment"][lane] and first["steps_by_environment"][lane]==1600 and [x["step"] for x in ss]==list(range(1000,1501,100)) and all(mx[i]<=v for i,v in enumerate([.15,.20,.20,.15])) and toe is not None and toe<12.)
  rows.append({"lane":lane,"mode":mode,"physical_failure":first["physical_failure_by_environment"][lane],"steps":first["steps_by_environment"][lane],"max_heading_error":mx[0],"max_lateral":mx[1],"max_speed_error":mx[2],"max_height_error":mx[3],"mean_height":statistics.mean(x[4] for x in vals) if len(vals)==6 else None,"max_toe_speed":toe,"pass":passed})
 groups=[]
 for mode in range(5):
  rr=[x for x in rows if x["mode"]==mode];heights=[x["mean_height"] for x in rr if x["mean_height"] is not None];groups.append({"mode":mode,"passing":sum(x["pass"] for x in rr),"height_median":statistics.median(heights) if len(heights)==8 else None})
 sep=groups[3]["height_median"]-groups[4]["height_median"] if all(groups[i]["height_median"] is not None for i in (3,4)) else None
 report={"argv":cmd,"shader_sha256":h(B/(arm+".metallib")),"library_sha256":h(D/"libmetalrobo.dylib"),"first_episode":first,"failed_environment_steps":data["failed_environment_steps"],"lanes":rows,"groups":groups,"climb_separation_m":sep,"screen_pass":all(x["pass"] for x in rows) and sep is not None and sep>=.25}
 (A/(arm+"-analysis.json")).write_text(json.dumps(report,indent=2)+"\n")
 with (B/(arm+".tsv")).open("rb") as f,(A/(arm+".tsv.gz")).open("wb") as raw:
  with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as gz:
   import shutil;shutil.copyfileobj(f,gz)
 print(arm,"passed",sum(x["pass"] for x in rows),"/40, separation",sep,"screen",report["screen_pass"],flush=True)
