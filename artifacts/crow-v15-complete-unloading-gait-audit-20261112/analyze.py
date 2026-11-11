from pathlib import Path
import gzip,importlib.util,json,math,os,subprocess
R=Path.cwd(); A=Path("artifacts/crow-v15-complete-unloading-gait-audit-20261112")
spec=importlib.util.spec_from_file_location("c","artifacts/crow-v15-corvid-stride-audit-20261111/analyze.py");c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
coarse=json.loads(Path("artifacts/crow-v15-small-ankle-delay-screen-20261112/coarse-comparison.json").read_text())
baseline=json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text())
datasets=[("reference",20261126,Path("artifacts/crow-v15-substep-gait-observer-20261111/autonomous-20261126")),("small-delay",20261126,Path("artifacts/crow-v15-small-ankle-delay-screen-20261112/autonomous-20261126")),("small-delay",20261127,Path("artifacts/crow-v15-small-ankle-delay-screen-20261112/autonomous-20261127"))]
out=[]
for label,seed,path in datasets:
 q={}; contacts={}; ends={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_Q_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);key=(e,t,k);assert key not in q;q[key]=list(map(float,z[4:]));assert len(q[key])==41
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4])
    if 201<=t<=300:
     key=(e,t,k);assert key not in contacts and int(z[4])<=int(z[5]);v=list(map(float,z[6:]));assert len(v)==37;contacts[key]=v
   elif l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);t,e=x["step"],x["lane"]
    if 201<=t<=300:ends[e,t]=x["q"]
 assert len(q)==len(contacts)==12800 and len(ends)==800
 assert max(abs(a-b) for e in range(8) for t in range(201,301) for a,b in zip(q[e,t,15],ends[e,t]))<1e-6
 inp="".join(f"{e} {t} {k} "+" ".join(map(str,v))+"\n" for (e,t,k),v in sorted(q.items()));env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")}
 p=subprocess.run(["build/crow-v15-substep-gait-observer-20261111/kinematics"],input=inp,text=True,capture_output=True,env=env,check=True);body={}
 for l in p.stdout.splitlines():
  z=l.split();e,t,k=map(int,z[1:4]);key=(e,t*16+k,z[4]);assert key not in body;body[key]=list(map(float,z[5:]))
 assert len(body)==12800*12
 for e in range(8):
  if label=="reference":
   allb=next(x for x in baseline["rows"] if x["seed"]==seed)["baseline"][e]["support_qualified_swing_bouts"]
  else:
   allb=[x["all_bouts"] for x in next(x for x in coarse["rows"] if x["seed"]==seed and x["lane"]==e)["sides"]]
  for si,side in enumerate(("left","right")):
   load=[contacts[e,201+n//16,n%16][2+si]>1e-7 for n in range(1600)]
   runs=[];start=None
   for n,loaded in enumerate(load+[True]):
    if not loaded and start is None:start=n
    if loaded and start is not None:
     lo,hi=start,n-1;start=None
     if lo==0 or hi==1599:continue
     overlapping=[b for b in allb[si] if any(lo<=(t-201)*16+15<=hi for t in range(b[0],b[1]+1))]
     if not overlapping:continue
     angles=[c.calc(body,q[e,201+i//16,i%16],e,(201+i//16)*16+i%16,side) for i in range(lo,hi+1)]
     ki=min(range(len(angles)),key=lambda j:angles[j][1]);ai=min(range(len(angles)),key=lambda j:angles[j][2])
     runs.append({"first_substep":[201+lo//16,lo%16],"last_substep":[201+hi//16,hi%16],"duration_ms":(hi-lo+1)*1.25,"overlapping_qualified_bouts":overlapping,"ankle_minus_knee_min_ms":(ai-ki)*1.25,"knee_min_substep":[201+(lo+ki)//16,(lo+ki)%16],"ankle_min_substep":[201+(lo+ai)//16,(lo+ai)%16],"knee_edge":ki in (0,len(angles)-1),"ankle_edge":ai in (0,len(angles)-1)})
   out.append({"dataset":label,"seed":seed,"lane":e,"side":side,"unloaded_intervals":runs})
 print(label,seed,"complete observed intervals",sum(len(x["unloaded_intervals"]) for x in out if x["dataset"]==label and x["seed"]==seed),flush=True)
result={"method":"Foot normal impulse threshold 1e-7 Ns/substep; bracketed by loaded samples; intervals overlap previously support-qualified endpoint-clear fragments. Extrema not extension onset.","rows":out,"limitations":"Development only; no proof of continuous clearance or contact-point slip; threshold/contact chatter can fragment swing; no crow reference angles."}
(A/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
