from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-flight-speed-feedback-20261113");P=A/"push-wing-20261126-rebound-speed";meta=json.loads((P/"run.stdout").read_text())["first_episode_navigation"];states={};targets={}
with gzip.open(P/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);states[x["lane"],x["step"]]=x
  elif line.startswith("CROW_DRIVE_TARGET "):
   z=line.split();t,e=map(int,z[1:3]);targets[e,t]=list(map(float,z[3:]))
def speed(x):
 q=x["q"];v=x["v"];xx,yy,zz,w=q[3:7];a=1-2*(yy*yy+zz*zz);b=2*(w*zz+xx*yy);return (a*v[0]+b*v[1])/math.hypot(a,b)
rows=[]
for e in range(2,8):
 end=min(meta["steps_by_environment"][e],900);gain=.01 if e<4 else .02 if e<6 else .04;opts=[]
 for offset in (-2,-1,0):
  errs=[];missing=[];active=0
  for t in range(435,end+1):
   if (e,t+offset) not in states or (e,t) not in targets:missing.append(t);continue
   val=-.04+gain*(.35-speed(states[e,t+offset]));expected=max(-.09,min(.025,val));err=max(abs(x-expected) for x in targets[e,t][6:8]);errs.append(err)
   if -.089<val<.024:active+=1
  opts.append({"offset":offset,"max_abs_error":max(errs) if errs else None,"unsaturated_samples":active,"missing_ticks":missing})
 valid=[x for x in opts if not x["missing_ticks"] and x["max_abs_error"] is not None and x["max_abs_error"]<=1e-4 and x["unsaturated_samples"]>=30]
 rows.append({"lane":e,"end":end,"timing_options":opts,"validated_offsets":[x["offset"] for x in valid],"target_speed_law_valid":bool(valid)})
audit=json.loads((A/"sensor-feedback-audit.json").read_text());main=json.loads((A/"analysis.json").read_text());negative=json.loads((A/"matched-negative-audit.json").read_text());pairs={}
for e in (2,4,6):
 key=str(e)+"-"+str(e+1);pairs[key]=bool(audit["mask_valid"] and audit["policy_pack_bytes_identical"] and main["paired_long_pass"][key] and all(next(x for x in rows if x["lane"]==k)["target_speed_law_valid"] and next(x for x in audit["rows"] if x["lane"]==k)["reference_parity_before_first_change"] for k in (e,e+1)) and (e!=4 or all(x["parity_before_first_target_difference"] for x in negative["rows"])))
out={"rows":rows,"validated_paired_exploratory_pass":pairs,"scope":"Full-blend targets reconstructed from accepted root quaternion and velocity. A pass is development screening only."};(A/"speed-target-audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
