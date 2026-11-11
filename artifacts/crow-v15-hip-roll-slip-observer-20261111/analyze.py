from pathlib import Path
import gzip,json,collections,subprocess,os,math,statistics
A=Path("artifacts/crow-v15-hip-roll-slip-observer-20261111");S=A/"gait-080-20261126";R=Path("artifacts/crow-v15-hip-roll-walk-launch-controls-20261111/gait-080-20261126");records={};counts=collections.Counter();diff=[]
def read(path):
 d={}
 with gzip.open(path,"rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);t=x["step"];e=x["lane"]
    if 201<=t<=301:d[e,t]=x
 return d
base=read(R/"run.stderr.gz");new=read(S/"run.stderr.gz")
for k in sorted(set(base)|set(new)):
 a=base.get(k);b=new.get(k)
 if a is None or b is None:diff.append([k,"missing"]);continue
 c=dict(b);c["contacts"]=[{key:value for key,value in z.items() if key!="point"} for z in b["contacts"]]
 if a!=c:diff.append([k,"state or contact mismatch"])
point_complete=all("point" in z and len(z["point"])==3 for x in new.values() for z in x["contacts"])
text="".join(str(e)+" "+str(t)+" "+" ".join(map(str,x["q"]+x["v"]))+"\n" for (e,t),x in sorted(new.items()))
env={**os.environ,"DYLD_LIBRARY_PATH":str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve())};p=subprocess.run(["build/crow-v15-hip-roll-slip-observer-20261111/kinematics"],input=text,text=True,capture_output=True,check=True,env=env);body={}
for l in p.stdout.splitlines():
 a=l.split()
 if a[0]=="body":body[int(a[1]),int(a[2]),a[3]]=list(map(float,a[4:]))
with gzip.open(S/"replay.json.gz","rt") as f:replay=json.load(f)["payload"]
assert replay["environment"]==1
frames={x["step"]:x for x in replay["frames"]};maxpos=maxrot=maxlin=maxang=maxq=0.;nvalid=0
names=["crow_"+side+"_foot" for side in ("left","right")]+["crow_"+side+"_"+d+"_"+seg for side in ("left","right") for d in ("front_toe_1","front_toe_2","front_toe_3","rear_toe") for seg in ("proximal","distal")]
for t in range(201,302):
 f=frames[t-1];maxq=max(maxq,max(abs(a-b) for a,b in zip(f["q"],new[1,t]["q"])))
 for name in names:
  z=body[1,t,name];j=replay["body_names"].index(name)*13;v=f["body_states"][j:j+13];maxpos=max(maxpos,math.dist(z[:3],v[:3]));maxrot=max(maxrot,min(math.dist(z[3:7],v[3:7]),math.dist(z[3:7],[-x for x in v[3:7]])));maxlin=max(maxlin,math.dist(z[7:10],v[7:10]));maxang=max(maxang,math.dist(z[10:13],v[10:13]));nvalid+=1
valid=maxq<1e-6 and maxpos<1e-5 and maxrot<1e-5 and maxlin<1e-4 and maxang<1e-4
vxyz=lambda a,b:[a[i]-b[i] for i in range(3)]
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
slips=collections.defaultdict(list)
for (e,t),sample in sorted(new.items()):
 for c in sample["contacts"]:
  a,b=c["a_name"],c["b_name"]
  if c["impulses"][0]<=1e-6 or (a!="locomotion_ground" and b!="locomotion_ground"):continue
  name=b if a=="locomotion_ground" else a
  if name not in names:continue
  z=body[e,t,name];r=vxyz(c["point"],z[:3]);rot=cross(z[10:13],r);vel=[z[7+i]+rot[i] for i in range(3)];normal=c["normal"];nn=math.sqrt(sum(v*v for v in normal));assert nn>.99;normal=[v/nn for v in normal];vn=sum(vel[i]*normal[i] for i in range(3));tangent=[vel[i]-vn*normal[i] for i in range(3)];slips[e].append(math.sqrt(sum(v*v for v in tangent)))
rows=[]
for e in range(8):
 v=sorted(slips[e]);rows.append({"lane":e,"loaded_contact_samples":len(v),"median_tangential_m_s":statistics.median(v) if v else None,"p95_tangential_m_s":v[math.ceil(.95*len(v))-1] if v else None,"maximum_tangential_m_s":max(v) if v else None})
out={"trajectory_and_existing_contact_equal":not diff and len(base)==len(new)==808,"trajectory_differences":diff[:20],"point_complete":point_complete,"kinematics_validation":{"passed":valid,"samples":nvalid,"q_max_error":maxq,"position_m":maxpos,"orientation":maxrot,"linear_velocity_m_s":maxlin,"angular_velocity_rad_s":maxang},"rows":rows,"limits":"Final-substep endpoint contact-point tangential speed only, weighted by contact count, not force. It does not bound slip within or between substeps. Ground is stationary; diagnostic on a selected seed, not a biological measurement."};(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
