from pathlib import Path
import collections,contextlib,gzip,importlib.util,io,json,math
A=Path("artifacts/crow-v15-release-perturbation-screen-20261112")
spec=importlib.util.spec_from_file_location("release_analysis",A/"analyze.py");m=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):spec.loader.exec_module(m)
out=[]
for e in (0,3):
 counts=collections.Counter();examples={}
 for t in range(451,551):
  v=m.by[e,t];x=m.state[e,t];q=x["q"];u=x["v"]
  tests={"left_support":v[0]>.0002,"right_support":v[1]>.0002,"combined_support":v[0]+v[1]>=.5*.45006*9.81*.02,"left_opposition":sum(v[21+i] for i in range(6))>=.0002 and sum(v[21+i] for i in (6,7))<=-.0002,"right_opposition":sum(v[29+i] for i in range(6))>=.0002 and sum(v[29+i] for i in (6,7))<=-.0002,"nonfoot_absent":abs(v[4])<=1e-7,"upright":math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))))<=.35,"slow":math.sqrt(sum(z*z for z in u[:3]))<=.12}
  for k,ok in tests.items():
   if not ok:
    counts[k]+=1
    if k not in examples:examples[k]={"step":t,"left_impulse":v[0],"right_impulse":v[1],"left_front_vertical_impulse":sum(v[21+i] for i in range(6)),"left_rear_vertical_impulse":sum(v[21+i] for i in (6,7)),"right_front_vertical_impulse":sum(v[29+i] for i in range(6)),"right_rear_vertical_impulse":sum(v[29+i] for i in (6,7))}
 caps=[]
 with gzip.open(A/"release/substeps.log.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_TOE_VELOCITY "):
    z=line.split();t,sub,lane=map(int,z[1:4]);v=list(map(float,z[-18:-2]))
    if lane==e and max(map(abs,v))>=11.99:caps.append({"step":t,"substep":sub,"toe_velocity_slot":max(range(len(v)),key=lambda i:abs(v[i])),"velocity":max(v,key=abs)})
 out.append({"lane":e,"last_100_intervals_failed_criteria":dict(counts),"first_examples":examples,"toe_cap_samples":caps})
result={"rows":out,"scope":"Development bench; logged impulses and sampled toe velocity, not proof of physical root cause. Counts are dependent interval observations."}
(A/"failure-diagnostics.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
