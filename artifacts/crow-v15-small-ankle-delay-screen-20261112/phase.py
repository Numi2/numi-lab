from pathlib import Path
import gzip,importlib.util,json,math,os,subprocess
R=Path.cwd();A=Path("artifacts/crow-v15-small-ankle-delay-screen-20261112");assert json.loads(Path("artifacts/crow-v15-substep-gait-observer-20261111/verification.json").read_text())["pass"]
spec=importlib.util.spec_from_file_location("c","artifacts/crow-v15-corvid-stride-audit-20261111/analyze.py");c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
coarse=json.loads((A/"coarse-comparison.json").read_text());rows=[]
for seed in (20261126,20261127):
 path=A/f"autonomous-{seed}";q={};endpoints={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);t,e=x["step"],x["lane"]
    if 201<=t<=300:endpoints[e,t]=x["q"]
   if l.startswith("CROW_Q_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);key=e,t,k
    if key in q:raise ValueError("duplicate q")
    v=list(map(float,z[4:]));assert len(v)==41 and all(map(math.isfinite,v));q[key]=v
 assert len(q)==12800 and len(endpoints)==800 and all((e,t,k) in q for e in range(8) for t in range(201,301) for k in range(16))
 assert max(abs(x-y) for e in range(8) for t in range(201,301) for x,y in zip(q[e,t,15],endpoints[e,t]))<1e-6
 inp="".join(f"{e} {t} {k} "+" ".join(map(str,v))+"\n" for (e,t,k),v in sorted(q.items()));env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")}
 p=subprocess.run(["build/crow-v15-substep-gait-observer-20261111/kinematics"],input=inp,text=True,capture_output=True,env=env,check=True);body={}
 for l in p.stdout.splitlines():
  z=l.split();e,t,k=map(int,z[1:4]);key=e,t*16+k,z[4];assert key not in body;body[key]=list(map(float,z[5:]))
 assert len(body)==12800*12
 for x in [v for v in coarse["rows"] if v["seed"]==seed]:
  e=x["lane"]
  for side in x["sides"]:
   values=[]
   for bout in side["interior"]:
    lo,hi=bout["bout"];ix=[(lo,15)]+[(t,k) for t in range(lo+1,hi+1) for k in range(16)];ang=[c.calc(body,q[e,t,k],e,t*16+k,side["side"]) for t,k in ix];ki=min(range(len(ang)),key=lambda i:ang[i][1]);ai=min(range(len(ang)),key=lambda i:ang[i][2]);values.append({"bout":[lo,hi],"ankle_minus_knee_min_ms":(ai-ki)*1.25,"edge_minimum":ki in (0,len(ix)-1) or ai in (0,len(ix)-1)})
   passing=bool(values) and sum(v["ankle_minus_knee_min_ms"]>=2.5 for v in values)>len(values)/2;rows.append({"seed":seed,"lane":e,"side":side["side"],"bouts":values,"phase_pass":passing})
count=sum(x["phase_pass"] for x in rows);out={"rows":rows,"side_lanes_passing":count,"lane_walking_gates_pass":coarse["all_lane_gates_pass"],"all_gates_pass":count>=24 and coarse["all_lane_gates_pass"],"limitations":"Selected endpoint-cleared intervals are not complete swings and clearance between endpoints is not proven; species-reference calibration and independent trials remain open."};(A/"comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=2))
