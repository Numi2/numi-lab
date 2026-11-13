from pathlib import Path
import gzip,json,os,subprocess,collections,hashlib
R=Path.cwd();d=R/"artifacts/crow-v15-launch-foot-location-audit-20261113";p=R/"artifacts/crow-v15-mirrored-leg-isolation-20261113/push-wing-20261126-rebound-mirrored/run.stderr.gz";exe=R/"build/crow-v15-mirrored-leg-isolation-20261113/kinematics";lib=exe.parent;states={};contacts=collections.defaultdict(list)
with gzip.open(p,"rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"]
   if e in (2,3) and 370<=t<=375:states[e,t]=x
  elif line.startswith("CROW_GROUND_CONTACT "):
   z=line.split();t,k,e=map(int,z[1:4]);
   if e in (2,3) and 371<=t<=375:contacts[e,t].append(z)
keys=sorted(states);assert len(keys)==12
inp="\n".join(str(i)+" 0 "+" ".join(map(str,states[k]["q"]+states[k]["v"])) for i,k in enumerate(keys))+"\n";pout=subprocess.run([str(exe)],input=inp,capture_output=True,text=True,check=True,env={**os.environ,"DYLD_LIBRARY_PATH":str(lib)}).stdout;com={}
for line in pout.splitlines():
 z=line.split();com[keys[int(z[1])]]=list(map(float,z[3:]))
rows=[]
for e in (2,3):
 for t in range(371,376):
  v=com[e,t-1]; cs=contacts[e,t];weight=sum(float(z[20]) for z in cs);assert weight>0
  center=sum(float(z[8])*float(z[20]) for z in cs)/weight
  rows.append({"lane":e,"tick":t,"preceding_control_state_com_x_m":v[1],"normal_impulse_weighted_ground_point_x_m":center,"ground_point_minus_preceding_com_x_m":center-v[1],"normal_impulse_ns":weight,"contact_records":len(cs)})
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={"rows":rows,"source_sha256":h(p),"kinematics_sha256":h(exe),"scope":"Posthoc coarse control-interval estimate using the previous accepted COM and contact points weighted by reported normal impulse; not instantaneous center of pressure or a proof of causation. Used-seed correlated lanes, modeled coordinates and forces, no biological calibration."};(d/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(rows,indent=2))
