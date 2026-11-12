from pathlib import Path
import importlib.util,json,math,collections
p=Path("artifacts/crow-v15-reduced-three-joint-lift-screen-20261112");s=importlib.util.spec_from_file_location("audit","artifacts/crow-v15-whole-gait-audit-20261111/audit.py");a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
ground=next(x["ground_z_m"] for x in json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text())["rows"] if x["seed"]==20261127)
paths={"held":Path("artifacts/crow-v15-small-support-stroke-20261112/sham-20261127-paired"),"three_joint_020":p/"sham-20261127-paired","three_joint_015":p/"candidate-20261127-rebound"};out={}
for label,path in paths.items():
 sup,sub,bodies,com=a.load(path);counts=collections.Counter();examples=[]
 for e in range(8):
  for t in range(201,301):
   for i,side in enumerate(("left","right")):
    vals=[]
    for name in a.names(side):
     z=bodies[e,t,name];x,y,zz,w=z[3:7];rot=(2*(x*zz-y*w),2*(y*zz+x*w),1-2*(x*x+y*y));half=(.010,.01725,.0092) if name.endswith("_foot") else (.009,.0025,.0092);vals.append(z[2]-ground-sum(abs(r)*h for r,h in zip(rot,half)))
    if vals[0]>.001 and min(vals[1:])<=.001:
     counts["total"]+=1;which=a.names(side)[1+vals[1:].index(min(vals[1:]))].replace("crow_"+side+"_","");counts["toe:"+which]+=1
     norm=sum(sub[e,t,k][2+i] for k in range(16));other=sum(sub[e,t,k][3-i] for k in range(16));phase=math.sin(2*math.pi*(t*.02+.005)/.5)*(1 if i==0 else -1)
     counts["loaded" if norm>1e-7 else "no_own_normal"]+=1;counts["opposite_supported" if other>=a.SUPPORT else "opposite_not_supported"]+=1;counts["positive_carrier" if phase>0 else "negative_carrier"]+=1
     if len(examples)<12:examples.append({"lane":e,"step":t,"side":side,"limiting_toe":which,"sole_gap_m":vals[0],"toe_gap_m":min(vals[1:]),"own_interval_normal_impulse":norm,"opposite_interval_normal_impulse":other,"carrier":phase})
 out[label]={"counts":dict(counts),"examples":examples}
out["scope"]="Endpoint geometric low-toe events joined to interval-summed foot normals; a loaded event may represent normal toe support and is not itself evidence of slip. Opposite support threshold is half body weight times 20 ms, not continuous support."
(p/"toe-phase.json").write_text(json.dumps(out,indent=2)+"\n")
for k in paths:print(k,out[k]["counts"])
