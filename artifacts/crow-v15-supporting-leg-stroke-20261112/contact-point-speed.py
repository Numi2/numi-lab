from pathlib import Path
import importlib.util,json,math,os,subprocess
p=Path("artifacts/crow-v15-supporting-leg-stroke-20261112");s=importlib.util.spec_from_file_location("a",p/"long-audit.py");a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
paths={"baseline":Path("artifacts/crow-v15-upper-body-transition-isolation-20261112/sham-20261127-held"),"stroke":p/"sham-20261127-stance"}
def cross(x,y):return (x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0])
def stat(v):
 v=sorted(v)
 return {"count":len(v),"median_m_s":v[(len(v)-1)//2] if v else None,"p95_m_s":v[int(.95*(len(v)-1))] if v else None,"max_m_s":max(v) if v else None}
def run(path):
 sup,sub,positions,com=a.load(path);v=[];excluded=0;parts={e:[] for e in range(8)}
 inp="".join(str(e)+" "+str(t)+" "+" ".join(map(str,x["q"]+x["v"]))+"\n" for (e,t),x in sorted(sup.items()))
 env={**os.environ,"DYLD_LIBRARY_PATH":str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve())}
 proc=subprocess.run(["build/crow-v15-gait-slip-observer-20261112/kinematics"],input=inp,text=True,capture_output=True,check=True,env=env);bodies={}
 for line in proc.stdout.splitlines():
  z=line.split()
  if z[0]=="body":
   vals=list(map(float,z[4:]));assert len(vals)==13 and all(math.isfinite(q) for q in vals);key=(int(z[1]),int(z[2]),z[3]);bodies[key]=vals
 assert all(key in bodies and math.dist(val[:3],bodies[key][:3])<1e-5 for key,val in positions.items())
 for (e,t),x in sup.items():
  for c in x["contacts"]:
   if c["impulses"][0]<=1e-7:continue
   name=c["a_name"] if c["b_name"]=="locomotion_ground" else c["b_name"] if c["a_name"]=="locomotion_ground" else ""
   if not name.startswith(("crow_left_","crow_right_")) or ("foot" not in name and "toe" not in name):continue
   if abs(c["normal"][2])<.99:excluded+=1;continue
   b=bodies[e,t,name];r=[c["point"][i]-b[i] for i in range(3)];w=cross(b[10:13],r);speed=math.hypot(b[7]+w[0],b[8]+w[1]);v.append(speed);parts[e].append(speed)
 return {"all":stat(v),"lanes":[{"lane":e,**stat(parts[e])} for e in range(8)],"nonvertical_excluded":excluded}
out={"arms":{k:run(v) for k,v in paths.items()},"scope":"Instantaneous tangential contact-point speed at recorded 20 ms endpoints and accepted normal contacts, not integrated slip distance or a biological measurement. Kinematic positions were independently replay-validated by compare.py."};(p/"contact-point-speed.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v["all"] for k,v in out["arms"].items()},indent=2))
