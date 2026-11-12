from pathlib import Path
import gzip,json,collections,math
p=Path("artifacts/crow-v15-loaded-toe-release-screen-20261112");paths={"sham":p/"sham-20261127-paired","candidate":p/"candidate-20261127-rebound"};arms={}
for name,path in paths.items():
 states={};sub={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"]
    if 200<=t<=300:states[e,t]=x
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if 201<=t<=300:sub[e,t,k]=list(map(float,z[6:]))
 assert len(states)==808 and len(sub)==12800
 rows=[]
 for e in range(8):
  intervals=[]
  for t in (201,226,251,276):
   v=[sub[e,t1,k] for t1 in range(t,t+25) for k in range(16)];start=states[e,t-1]["q"];end=states[e,t+24]["q"]
   intervals.append({"steps":[t,t+24],"forward_displacement_m":end[0]-start[0],"unsupported_substeps":sum(a[2]<=1e-7 and a[3]<=1e-7 for a in v),"toe_normal_impulse":[sum(sum(a[5:13]) for a in v),sum(sum(a[13:21]) for a in v)],"foot_total_normal_impulse":[sum(a[2] for a in v),sum(a[3] for a in v)]})
  rows.append({"lane":e,"cycles":intervals})
 arms[name]=rows
out={"arms":arms,"scope":"Post-hoc .5-second blocks on development seed. Root displacement and normal impulses are simulated; the unsupported threshold is not a fall detector; a block is a controller period, not necessarily a valid biological stride."};(p/"cycle-diagnostics.json").write_text(json.dumps(out,indent=2)+"\n")
for i in range(8):
 print("lane",i,"sham mm",[round(1000*x["forward_displacement_m"],1) for x in arms["sham"][i]["cycles"]],"candidate mm",[round(1000*x["forward_displacement_m"],1) for x in arms["candidate"][i]["cycles"]],"unloaded",[x["unsupported_substeps"] for x in arms["sham"][i]["cycles"]],"->",[x["unsupported_substeps"] for x in arms["candidate"][i]["cycles"]])
