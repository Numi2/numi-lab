from pathlib import Path
import collections,gzip,importlib.util,json,math,os,subprocess
A=Path("artifacts/crow-v15-substep-gait-observer-20261111");P=A/"autonomous-20261126";R=Path.cwd()
assert json.loads((A/"verification.json").read_text())["pass"]
spec=importlib.util.spec_from_file_location("angles","artifacts/crow-v15-corvid-stride-audit-20261111/analyze.py");c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
q={};contacts={}
with gzip.open(P/"run.stderr.gz","rt") as f:
 for l in f:
  z=l.split()
  if not z or z[0] not in ("CROW_Q_SUBSTEP","CROW_TWO_TOE_SUBSTEP"):continue
  t,k,e=map(int,z[1:4])
  if not 201<=t<=300:continue
  key=e,t,k
  if z[0]=="CROW_Q_SUBSTEP":q[key]=list(map(float,z[4:]))
  else:contacts[key]=list(map(float,z[6:]))
assert len(q)==len(contacts)==12800
inp="".join(f"{e} {t} {k} "+" ".join(map(str,v))+"\n" for (e,t,k),v in sorted(q.items()))
env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")};p=subprocess.run(["build/crow-v15-substep-gait-observer-20261111/kinematics"],input=inp,text=True,capture_output=True,env=env,check=True)
body={}
for l in p.stdout.splitlines():
 z=l.split();e,t,k=map(int,z[1:4]);v=list(map(float,z[5:]));assert len(v)==7 and all(map(math.isfinite,v));key=(e,t*16+k,z[4]);assert key not in body;body[key]=v
assert len(body)==12800*12
old=json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text());rows=next(x["baseline"] for x in old["rows"] if x["seed"]==20261126)
results=[]
for e in range(8):
 for si,side in enumerate(("left","right")):
  boutrows=[]
  for lo,hi in rows[e]["support_qualified_swing_bouts"][si]:
   if lo<=201 or hi>=300:continue
   indices=[(lo,15)]+[(t,k) for t in range(lo+1,hi+1) for k in range(16)];angles=[c.calc(body,q[e,t,k],e,t*16+k,side) for t,k in indices];mi=[min(range(len(angles)),key=lambda j:angles[j][i]) for i in range(3)];delta=(mi[2]-mi[1])*.00125
   boutrows.append({"endpoint_bounds":[lo,hi],"q_samples":len(indices),"min_sample_index_hip_knee_ankle":mi,"ankle_min_after_knee_min_ms":delta*1000,"knee_or_ankle_min_at_bout_edge":mi[1] in (0,len(indices)-1) or mi[2] in (0,len(indices)-1)})
  # Normal impulse over individual substeps, not established contact kinematics.
  toe=[]
  for t in range(201,301):
   for k in range(16):
    x=contacts[e,t,k];toe.append([x[5+8*si+6]+x[5+8*si+7]>1e-7,x[5+8*si+2]+x[5+8*si+3]>1e-7])
  results.append({"lane":e,"side":side,"bouts":boutrows,"hallux_substeps_with_normal_impulse":sum(x[0] for x in toe),"middle_front_toe_substeps_with_normal_impulse":sum(x[1] for x in toe)})
allb=[b for x in results for b in x["bouts"]];d=[b["ankle_min_after_knee_min_ms"] for b in allb];counts=collections.Counter("knee_first" if x>0 else "ankle_first" if x<0 else "same_substep" for x in d)
out={"seed":20261126,"rows":results,"bout_count":len(allb),"minimum_order_counts":counts,"edge_censored_count":sum(b["knee_or_ankle_min_at_bout_edge"] for b in allb),"ankle_minus_knee_min_ms_range":[min(d),max(d)],"limitations":"Only previously qualified endpoint-cleared intervals, not complete swing or confirmed continuous clearance. Extrema outside these intervals are not observed. Two distinct control-development seeds were previously tested but this instrument runs one. This is not American-crow kinematics."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=2))
