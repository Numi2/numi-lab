from pathlib import Path
import gzip,hashlib,json,math,os,subprocess
A=Path("artifacts/crow-v15-lift-horizontal-diagnostic-20261112")
src=Path("artifacts/crow-v15-small-support-stroke-20261112/candidate-20261127-rebound/run.stderr.gz")
offsets=(0.,.1,.2,.3)
states={}
with gzip.open(src,"rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   z=json.loads(line.split(" ",1)[1]);t=z["step"]
   if z["lane"]==5 and t in range(225,301,5):states[t]=z
assert len(states)==16
requests=[];mapping={}
for t,z in sorted(states.items()):
 for side,idx in (("left",15),("right",18)):
  for delta in offsets:
   n=len(mapping);mapping[n]=(t,side,delta);q=z["q"].copy();q[idx-1]-=.5*delta;q[idx]+=delta;q[idx+1]-=.5*delta
   requests.append(str(n)+" "+str(t)+" "+" ".join(map(str,q+z["v"])))
kin=Path("build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics")
env=dict(os.environ,DYLD_LIBRARY_PATH=str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve()))
r=subprocess.run([str(kin)],input="\n".join(requests)+"\n",capture_output=True,text=True,check=True,env=env)
ground=next(x["ground_z_m"] for x in json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text())["rows"] if x["seed"]==20261127)
vals={i:{} for i in mapping}
for line in r.stdout.splitlines():
 z=line.split();n=int(z[1]);side=mapping[n][1];name=z[3]
 names=["crow_"+side+"_foot"]+["crow_"+side+"_"+d+"_"+seg for d in ("front_toe_1","front_toe_2","front_toe_3","rear_toe") for seg in ("proximal","distal")]
 if z[0]!="body" or name not in names:continue
 v=list(map(float,z[4:]));x,y,zz,w=v[3:7];rot=(2*(x*zz-y*w),2*(y*zz+x*w),1-2*(x*x+y*y));half=(.010,.01725,.0092) if name.endswith("_foot") else (.009,.0025,.0092)
 vals[n][name]=v[2]-ground-sum(abs(a)*b for a,b in zip(rot,half))
assert all(len(v)==9 for v in vals.values())
rows=[]
for n,(t,side,delta) in mapping.items():
 v=vals[n];foot=v["crow_"+side+"_foot"];toes={k:x for k,x in v.items() if not k.endswith("_foot")};minname=min(toes,key=toes.get)
 rows.append({"step":t,"side":side,"paired_knee_offset_rad":delta,"sole_gap_m":foot,"minimum_toe_gap_m":toes[minname],"limiting_toe":minname,"all_foot_minimum_gap_m":min(v.values())})
out={"source":str(src),"source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"kinematics_sha256":hashlib.sha256(kin.read_bytes()).hexdigest(),"ground_z_m":ground,"rows":rows,"scope":"Lane 5, 16 saved endpoint poses, torso and remaining joints fixed; hip and ankle each change by minus half of the knee change; geometric perturbations are not actuator targets, dynamics, reachability, or biological data."}
(A/"hip-paired-geometry.json").write_text(json.dumps(out,indent=2)+"\n")
for side in ("left","right"):
 print(side)
 for t in sorted(states):
  rr=[x for x in rows if x["step"]==t and x["side"]==side]
  print(t," ".join("%+.1f:%+.1f/%s"%(x["paired_knee_offset_rad"],1000*x["all_foot_minimum_gap_m"],x["limiting_toe"].replace("crow_"+side+"_", "")) for x in rr))
