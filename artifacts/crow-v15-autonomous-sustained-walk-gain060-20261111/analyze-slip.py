from pathlib import Path
import gzip,json,math,os,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-autonomous-sustained-walk-gain060-20261111";D=A/"autonomous-20261126";s={}
with gzip.open(D/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);
   if 201<=x["step"]<=600:s[x["lane"],x["step"]]=x
assert len(s)==8*400
text="".join(str(e)+" "+str(t)+" "+" ".join(map(str,x["q"]+x["v"]))+"\n" for (e,t),x in sorted(s.items()))
env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")}
p=subprocess.run([str(R/"build/crow-v15-autonomous-sustained-walk-gain060-20261111/slip-kinematics")],input=text,text=True,capture_output=True,check=True,env=env);bodies={}
for line in p.stdout.splitlines():
 z=line.split()
 if z[0]=="body":bodies[int(z[1]),int(z[2]),z[3]]=list(map(float,z[4:]))
with gzip.open(D/"replay.json.gz","rt") as f:replay=json.load(f)["payload"]
assert replay["environment"]==5 and replay["body_state_stride"]==13
names=replay["body_names"];frames={f["step"]:f for f in replay["frames"] if 200<=f["step"]<=599};assert len(frames)==400
errors=[0.,0.,0.]
for t,f in frames.items():
 for name in names:
  if not name.startswith("crow_") or (5,t+1,name) not in bodies:continue
  state=f["body_states"][names.index(name)*13:names.index(name)*13+13];kin=bodies[5,t+1,name]
  for j,sl in enumerate((slice(0,3),slice(7,10),slice(10,13))):errors[j]=max(errors[j],max(abs(a-b) for a,b in zip(state[sl],kin[sl])))
assert max(errors)<1e-4,errors
out=[];points=0;heights=[]
for lane in range(8):
 speeds=[];byfoot={"left":[],"right":[]}
 for t in range(201,601):
  for c in s[lane,t]["contacts"]:
   if "locomotion_ground" not in (c["a_name"],c["b_name"]) or c["impulses"][0]<=1e-7:continue
   name=c["b_name"] if c["a_name"]=="locomotion_ground" else c["a_name"]
   if not (name.startswith("crow_left_") or name.startswith("crow_right_")) or ("foot" not in name and "toe" not in name):continue
   assert "point" in c,(lane,t,name);points+=1;pos=c["point"];heights.append(pos[2]);q=bodies[lane,t,name];d=[pos[i]-q[i] for i in range(3)];w=q[10:13];v=q[7:10];cross=[w[1]*d[2]-w[2]*d[1],w[2]*d[0]-w[0]*d[2],w[0]*d[1]-w[1]*d[0]];u=[v[i]+cross[i] for i in range(3)];n=c["normal"];length=math.sqrt(sum(z*z for z in n));assert length>.9;n=[z/length for z in n];dot=sum(a*b for a,b in zip(u,n));speed=math.sqrt(sum((u[i]-dot*n[i])**2 for i in range(3)));speeds.append(speed);byfoot["left" if "_left_" in name else "right"].append(speed)
 def stats(v):
  v=sorted(v);return {"samples":len(v),"median_m_s":v[len(v)//2] if v else None,"p95_m_s":v[min(len(v)-1,int(.95*len(v)))] if v else None,"maximum_m_s":max(v) if v else None}
 out.append({"lane":lane,"all":stats(speeds),"feet":{k:stats(v) for k,v in byfoot.items()}})
assert points>0 and max(abs(z) for z in heights)<.10,(points,min(heights),max(heights))
x={"lane5_replay_max_abs_error":{"position_m":errors[0],"linear_velocity_m_s":errors[1],"angular_velocity_rad_s":errors[2]},"point_height_range_m":[min(heights),max(heights)],"rows":out,"scope":"Tangential velocity at the logged accepted endpoint contact points from independent articulated q/v kinematics, on positive-impulse ground contacts; no continuous or substep slip bound. Contact point is from the final substep and may precede accepted-state kinematics. Observer parity must be checked separately. One used seed/layout, not biological validation."};(A/"slip-analysis.json").write_text(json.dumps(x,indent=2)+"\n");print(json.dumps(x,indent=2))
