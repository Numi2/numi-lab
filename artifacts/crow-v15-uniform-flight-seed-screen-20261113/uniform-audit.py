from pathlib import Path
import gzip,json,math,sys
A=Path("artifacts/crow-v15-uniform-flight-seed-screen-20261113");D=A/sys.argv[1];gain=.08 if "push-wing-008-" in D.name else .12;isreference="push-wing-008-20261126-" in D.name;Q=Path("artifacts/crow-v15-flight-speed-gain-screen-20261113/push-wing-20261126-rebound-gain");meta=json.loads((D/"run.stdout").read_text())["first_episode_navigation"];analysis=json.loads((D/"analysis.json").read_text());data=[]
for p in ((Q,D) if isreference else (D,)):
 d={"state":{},"target":{},"mask":[],"effort":{},"errors":[],"duplicate":[]};data.append(d)
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);key=(x["lane"],x["step"])
    if key in d["state"]:d["duplicate"].append(["state",key])
    d["state"][key]=x;continue
   z=line.split()
   if not z:continue
   if z[0]=="CROW_SCENE_MASK":d["mask"].append((z[1],int(z[2]),int(z[3])))
   if "OBSERVER_ERROR" in z[0]:d["errors"].append(line.strip())
   if z[0]=="CROW_DRIVE_TARGET":t,e=map(int,z[1:3]);d["target"][e,t]=list(map(float,z[3:]))
   if z[0]=="CROW_WING_EFFORT":t,k,e=map(int,z[1:4]);d["effort"][e,t,k]=list(map(float,z[4:]))
new=data[-1];old=data[0] if isreference else None
exp={"crow_course_gate_left/collision","crow_course_gate_right/collision","crow_course_slalom_a/collision","crow_course_slalom_b/collision","crow_course_perch/collision"};course=[x for x in new["mask"] if x[0].startswith("crow_course_")];ground=[x for x in new["mask"] if x[0]=="locomotion_ground/collision"];maskvalid=set(x[0] for x in course)==exp and len(course)==5 and all(x[1]!=0 and x[2]==0 for x in course) and len(ground)==1 and ground[0][1]==ground[0][2] and ground[0][2]!=0 and all(x[1]==x[2] for x in new["mask"] if x not in course)
a=json.loads((D/"inputs.json").read_text());b=json.loads((Q/"inputs.json").read_text());policyvalid=all(a[k]==b[k] for k in ("landing_policy_sha256","route_policy_sha256"))
def speed(x):
 q=x["q"];v=x["v"];xx,yy,zz,w=q[3:7];u=1-2*(yy*yy+zz*zz);v0=2*(w*zz+xx*yy);return (u*v[0]+v0*v[1])/math.hypot(u,v0)
rows=[]
for e in range(2,8):
 end=min(meta["steps_by_environment"][e],900);opts=[]
 for off in (-2,-1,0):
  err=[];missing=[];active=0
  for t in range(435,end+1):
   if (e,t+off) not in new["state"] or (e,t) not in new["target"]:missing.append(t);continue
   raw=-.04+gain*(.35-speed(new["state"][e,t+off]));wanted=max(-.09,min(.025,raw));err.append(max(abs(x-wanted) for x in new["target"][e,t][6:8]));active+=int(-.089<raw<.024)
  opts.append({"offset":off,"max_error":max(err) if err else None,"unsaturated_samples":active,"missing":missing})
 valid=[x["offset"] for x in opts if not x["missing"] and x["max_error"] is not None and x["max_error"]<=1e-4 and x["unsaturated_samples"]>=30]
 effkeys=[(e,t,k) for t in range(348,501) for k in range(16)];effok=all(k in new["effort"] and len(new["effort"][k])==14 and all(math.isfinite(x) for x in new["effort"][k]) and abs(new["effort"][k][0])<=3.00001 and abs(new["effort"][k][7])<=3.00001 for k in effkeys)
 parity=None
 if isreference and e in (4,5):
  oldmeta=json.loads((Q/"run.stdout").read_text())["first_episode_navigation"];oldend=oldmeta["steps_by_environment"][e];parity=oldend==end and all((e,t) in old["state"] and (e,t) in new["state"] and all(old["state"][e,t][k]==new["state"][e,t][k] for k in ("q","v","policy_action","contacts")) for t in range(1,end+1))
 main=next(x for x in analysis["rows"] if x["lane"]==e);passed=bool(main["exploratory_long_screen_pass"] and valid and effok and not new["errors"] and not new["duplicate"] and maskvalid and policyvalid and (parity is not False))
 rows.append({"lane":e,"main_gate":main["exploratory_long_screen_pass"],"target_timing_options":opts,"validated_offsets":valid,"observed_flap_effort_bounds_348_500":effok,"whole_episode_reference_parity":parity,"pass":passed})
out={"name":D.name,"gain":gain,"mask_valid":maskvalid,"policy_bytes_match":policyvalid,"observer_errors":new["errors"],"duplicates":new["duplicate"],"rows":rows,"all_six_candidate_lanes_pass":all(x["pass"] for x in rows),"scope":"Used-seed obstacle-free screen. Flap effort after tick 500 is unobserved. Passing is not flight qualification."};(D/"uniform-audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
