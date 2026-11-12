from pathlib import Path
import collections,gzip,json,math,sys
A=Path("artifacts/crow-v15-continuous-leg-release-screen-20261112");D=A/("candidate-"+sys.argv[1]+"-rebound-hip");meta=json.loads((D/"run.stdout").read_text())["first_episode_navigation"];ends=meta["steps_by_environment"];mapping={}
for line in Path("artifacts/crow-v15-drive-work-observer-20261112/dof-map.txt").read_text().splitlines():
 z=line.split();mapping[int(z[0])]={"name":z[1],"limit_rad_s":float(z[-1])}
groups=collections.defaultdict(lambda:collections.defaultdict(lambda:{"peak_rad_s":0.,"hits":0,"examples":[]}));count=collections.Counter();errors=[]
with gzip.open(D/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_DRIVE_WORK "):
   z=line.split();t,k,e,j,flags,qi=map(int,z[1:7]);v0,v1=map(float,z[8:10]);
   if t>ends[e]:continue
   assert 535<=t<=625 and math.isfinite(v0) and math.isfinite(v1);count[e]+=1;lim=mapping[j]["limit_rad_s"]
   if lim>0 and flags&8:
    x=groups[e][j];x["peak_rad_s"]=max(x["peak_rad_s"],abs(v0),abs(v1))
    if abs(v1)>=lim-1e-4:x["hits"]+=1; x["examples"].append([t,k]) if len(x["examples"])<10 else None
  elif line.startswith("CROW_SUBSTEP_ERROR"):errors.append(line.strip())
rows=[]
for e in range(8):
 expected=max(0,min(ends[e],625)-534)*16*40
 rows.append({"lane":e,"drive_records":count[e],"expected":expected,"complete":count[e]==expected,"speed_limited_joints":[{"index":j,**mapping[j],**x} for j,x in sorted(groups[e].items()) if x["hits"]],"all_joint_limit_hits":sum(x["hits"] for x in groups[e].values())})
r={"rows":rows,"errors":errors,"scope":"Authored joint speed limits, endpoint velocities at each substep during steps 535–625 only; not biological speed limits or full-flight observation."};(A/("joint-speed-"+sys.argv[1]+".json")).write_text(json.dumps(r,indent=2)+"\n")
for x in rows:print(x)
print("errors",errors)
