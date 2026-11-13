from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-flight-pitch-feedback-20261113");P=A/"push-wing-20261126-rebound-pitch";meta=json.loads((P/"run.stdout").read_text())["first_episode_navigation"];states={};targets={}
with gzip.open(P/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);states[x["lane"],x["step"]]=x
  elif line.startswith("CROW_DRIVE_TARGET "):
   z=line.split();t,e=map(int,z[1:3]);targets[e,t]=list(map(float,z[3:]))
def pitch(q):
 x,y,z,w=q[3:7];return math.asin(max(-1.,min(1.,2*(w*y-z*x))))
rows=[]
for e in range(2,8):
 end=min(meta["steps_by_environment"][e],900);gain=.2 if e<4 else .35 if e<6 else .5;opts=[]
 for offset in (-2,-1,0):
  errs=[];rawerrs=[];missing=[];active=0
  for t in range(435,end+1):
   if (e,t+offset) not in states or (e,t) not in targets:missing.append(t);continue
   q=states[e,t+offset]["q"];v=.12+gain*(.05-pitch(q));expected=max(0.,min(.16,v));target=targets[e,t][6:8]
   err=max(abs(x-expected) for x in target);errs.append(err)
   if .001<v<.159:active+=1;rawerrs.append(err)
  opts.append({"offset":offset,"max_abs_error":max(errs) if errs else None,"max_unsaturated_error":max(rawerrs) if rawerrs else None,"unsaturated_samples":active,"missing_ticks":missing})
 valid=[x for x in opts if not x["missing_ticks"] and x["max_abs_error"] is not None and x["max_abs_error"]<=1e-4 and x["unsaturated_samples"]>=30]
 rows.append({"lane":e,"end":end,"timing_options":opts,"validated_offsets":[x["offset"] for x in valid],"target_pitch_law_valid":bool(valid)})
audit=json.loads((A/"sensor-feedback-audit.json").read_text());main=json.loads((A/"analysis.json").read_text());pairs={}
for e in (2,4,6):
 key=str(e)+"-"+str(e+1);pairs[key]=bool(audit["mask_valid"] and audit["policy_pack_bytes_identical"] and main["paired_long_pass"][key] and all(next(x for x in rows if x["lane"]==k)["target_pitch_law_valid"] and next(x for x in audit["rows"] if x["lane"]==k)["reference_parity_before_first_change"] for k in (e,e+1)))
out={"rows":rows,"validated_paired_exploratory_pass":pairs,"scope":"Reconstructs full-blend sweep drive targets from accepted root orientation, checking offsets and requiring unsaturated samples. Does not qualify flight."};(A/"pitch-target-audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
