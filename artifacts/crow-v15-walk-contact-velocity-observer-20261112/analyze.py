from pathlib import Path
import gzip,json,math,os,subprocess
R=Path.cwd();A=Path("artifacts/crow-v15-walk-contact-velocity-observer-20261112");P=A/"autonomous-20261126"
assert json.loads((A/"verification.json").read_text())["pass"]
states={}
with gzip.open(P/"run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(l.split(" ",1)[1]);e,t=x["lane"],x["step"]
   if 201<=t<=300:assert (e,t) not in states;states[e,t]=x
assert len(states)==800
inp="".join(f"{e} {t} "+" ".join(map(str,x["q"]+x["v"]))+chr(10) for (e,t),x in sorted(states.items()))
env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")}
p=subprocess.run(["build/crow-v15-walk-contact-velocity-observer-20261112/kinematics"],input=inp,text=True,capture_output=True,check=True,env=env);body={}
for l in p.stdout.splitlines():
 z=l.split();key=int(z[1]),int(z[2]),z[3];assert key not in body;body[key]=list(map(float,z[4:]))
with gzip.open(P/"replay.json.gz","rt") as f:r=json.load(f)["payload"]
assert r["environment"]==5 and r["body_state_stride"]==13
frames={f["step"]+1:f for f in r["frames"] if 200<=f["step"]<=299}
assert len(frames)==100
err={"position_m":0.,"orientation":0.,"linear_velocity_m_s":0.,"angular_velocity_rad_s":0.,"q":0.,"v":0.};checked=0
for t,f in frames.items():
 err["q"]=max(err["q"],max(abs(a-b) for a,b in zip(states[5,t]["q"],f["q"])))
 err["v"]=max(err["v"],max(abs(a-b) for a,b in zip(states[5,t]["v"],f["v"])))
 for name in r["body_names"]:
  if (5,t,name) not in body:continue
  i=r["body_names"].index(name)*13;y=f["body_states"][i:i+13];x=body[5,t,name];checked+=1
  err["position_m"]=max(err["position_m"],math.dist(x[:3],y[:3]))
  err["orientation"]=max(err["orientation"],min(math.dist(x[3:7],y[3:7]),math.dist(x[3:7],[-v for v in y[3:7]])))
  err["linear_velocity_m_s"]=max(err["linear_velocity_m_s"],math.dist(x[7:10],y[7:10]))
  err["angular_velocity_rad_s"]=max(err["angular_velocity_rad_s"],math.dist(x[10:13],y[10:13]))
passed=checked==1800 and err["q"]<1e-6 and err["v"]<1e-6 and err["position_m"]<1e-5 and err["orientation"]<1e-5 and err["linear_velocity_m_s"]<1e-4 and err["angular_velocity_rad_s"]<1e-4
out={"kinematics_validation":{"pass":passed,"checked_body_frames":checked,"max_errors":err},"rows":[],"limits":"Instantaneous accepted-endpoint estimate from solver contact point and post-step native body velocities; normal impulse threshold 1e-7 N s per contact, fixed ground assumed. Contact point and post-step velocity synchronization may differ within the final substep. Not integrated slip distance or biological validation."}
if passed:
 for e in range(8):
  for side in ("left","right"):
   weighted=0.;weights=0.;vmax=0.;n=0;loaded=0
   for t in range(201,301):
    touched=False
    for contact in states[e,t]["contacts"]:
     name=contact["a_name"] if contact["b_name"]=="locomotion_ground" else contact["b_name"] if contact["a_name"]=="locomotion_ground" else ""
     if not name.startswith("crow_"+side+"_") or not ("foot" in name or "toe" in name):continue
     impulse=contact["impulses"][0]
     if impulse<=1e-7:continue
     if (e,t,name) not in body:raise ValueError("missing body "+name)
     x=body[e,t,name];r=[contact["point"][i]-x[i] for i in range(3)];w=x[10:13];v=x[7:10];cross=[w[1]*r[2]-w[2]*r[1],w[2]*r[0]-w[0]*r[2],w[0]*r[1]-w[1]*r[0]];v=[v[i]+cross[i] for i in range(3)];normal=contact["normal"];norm=math.sqrt(sum(k*k for k in normal));assert .999<norm<1.001
     d=sum(v[i]*normal[i] for i in range(3));speed=math.sqrt(max(0.,sum(vv*vv for vv in v)-d*d))
     weighted+=speed*impulse;weights+=impulse;vmax=max(vmax,speed);n+=1;touched=True
    loaded+=touched
   out["rows"].append({"lane":e,"side":side,"loaded_endpoints":loaded,"loaded_contact_samples":n,"impulse_sum_Ns":weights,"impulse_weighted_tangent_speed_m_s":weighted/weights if weights else None,"max_tangent_speed_m_s":vmax if n else None})
(A/"analysis.json").write_text(json.dumps(out,indent=2)+chr(10));print(json.dumps(out,indent=2))
