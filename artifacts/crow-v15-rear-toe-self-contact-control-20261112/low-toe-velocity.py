from pathlib import Path
import collections,gzip,json,math,os,subprocess,statistics
p=Path("artifacts/crow-v15-rear-toe-self-contact-control-20261112");src=Path("artifacts/crow-v15-rear-toe-self-contact-control-20261112")
paths={"held":Path("artifacts/crow-v15-small-support-stroke-20261112/sham-20261127-paired"),"attached":src/"candidate-20261127"}
ground=next(x["ground_z_m"] for x in json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text())["rows"] if x["seed"]==20261127)
def names(side):return ["crow_"+side+"_foot"]+["crow_"+side+"_"+d+"_"+s for d in ("front_toe_1","front_toe_2","front_toe_3","rear_toe") for s in ("proximal","distal")]
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def stats(v):
 v=sorted(v)
 return {"count":len(v),"median_m_s":statistics.median(v) if v else None,"p95_m_s":v[int(.95*(len(v)-1))] if v else None,"over_0_01_fraction":sum(x>.01 for x in v)/len(v) if v else None,"max_m_s":max(v) if v else None}
out={}
for label,path in paths.items():
 st={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
    if 201<=t<=300:st[e,t]=x
 assert len(st)==800
 inp="".join(str(e)+" "+str(t)+" "+" ".join(map(str,x["q"]+x["v"]))+"\n" for (e,t),x in sorted(st.items()))
 env={**os.environ,"DYLD_LIBRARY_PATH":str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve())};proc=subprocess.run(["build/crow-v15-rear-toe-attachment-contact-20261112/kinematics-velocity" if label=="attached" else "build/crow-v15-gait-slip-observer-20261112/kinematics"],input=inp,text=True,capture_output=True,check=True,env=env);bodies={}
 for line in proc.stdout.splitlines():
  z=line.split()
  if z[0]=="body":
   v=list(map(float,z[4:]));assert len(v)==13;bodies[int(z[1]),int(z[2]),z[3]]=v
 counts=collections.Counter();velocities=[];per_side=collections.Counter();samples=[];contact_phases=collections.Counter();impulses=[];contact_parts=collections.Counter()
 for (e,t),x in st.items():
  for side in ("left","right"):
   boxes=names(side);gaps=[]
   for name in boxes:
    b=bodies[e,t,name];qx,qy,qz,qw=b[3:7];rot=(2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy));half=(.010,.01725,.0092) if name.endswith("_foot") else (.009,.0025,.0092);gaps.append(b[2]+(rot[0]*0.000650716945529 if label=="attached" and name.endswith("_foot") else 0.0)-ground-sum(abs(q)*h for q,h in zip(rot,half)))
   if not (gaps[0]>.001 and min(gaps[1:])<=.001):continue
   counts["low_toe_endpoints"]+=1;per_side[side]+=1;contacts=[]
   for c in x["contacts"]:
    if c["impulses"][0]<=1e-7:continue
    n=c["a_name"] if c["b_name"]=="locomotion_ground" else c["b_name"] if c["a_name"]=="locomotion_ground" else ""
    if n not in boxes[1:] or abs(c["normal"][2])<.99:continue
    b=bodies[e,t,n];r=[c["point"][j]-b[j] for j in range(3)];w=cross(b[10:13],r);v=[b[7+j]+w[j] for j in range(3)];speed=math.hypot(v[0],v[1]);contacts.append(speed);velocities.append(speed);impulses.append(c["impulses"][0]);contact_parts[n.replace("crow_"+side+"_","")]+=1;phase=math.sin(2*math.pi*(t*.02+.005)/.5)*(1 if side=="left" else -1);contact_phases["positive_carrier" if phase>0 else "negative_carrier"]+=1
   counts["with_accepted_toe_contact" if contacts else "without_accepted_toe_contact"]+=1
   if contacts and len(samples)<8:samples.append({"lane":e,"step":t,"side":side,"contact_speeds_m_s":contacts,"lowest_toe_gap_m":min(gaps[1:])})
 out[label]={"counts":dict(counts),"sides":dict(per_side),"contact_speed":stats(velocities),"contact_normal_impulses":{"count":len(impulses),"median":statistics.median(impulses) if impulses else None,"max":max(impulses) if impulses else None},"contact_parts":dict(contact_parts),"contact_phases":dict(contact_phases),"samples":samples}
out["scope"]="Previously used seed; 100 endpoints per lane. Foot-box and toe-box gap and positive accepted endpoint toe contacts are evaluated at the same logged state. Contact point speeds use native FK world linear/angular velocities and stored points. These sampled contacts are not integrated slip, kinetic friction, stance duty factor, or biological validation; low gaps without accepted contact are not demonstrated dragging."
(p/"low-toe-velocity.json").write_text(json.dumps(out,indent=2)+"\n")
for label in paths:print(label,out[label]["counts"],out[label]["contact_speed"])
