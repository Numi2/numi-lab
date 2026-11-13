from pathlib import Path
import gzip,json,math,statistics,sys
P=Path("artifacts/crow-v15-leg-flight-height-switch-20261113")/sys.argv[1];meta=json.loads((P/"run.stdout").read_text())["first_episode_navigation"];states={};targets={}
with gzip.open(P/"run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(l.split(" ",1)[1]);e,t=x["lane"],x["step"]
   if e in (6,7) and t<=meta["steps_by_environment"][e]:states[e,t]=x
  elif l.startswith("CROW_DRIVE_TARGET "):
   z=l.split();t,e=map(int,z[1:3]);
   if e in (6,7) and t<=meta["steps_by_environment"][e]:targets[e,t]=list(map(float,z[3:]))
rows=[]
for e in (6,7):
 first=.55 if e==6 else .80;final=.80 if e==6 else .55;windows=[]
 for lo,hi,goal in ((1000,1150,first),(1550,1800,final)):
  vals=[states[e,t]["q"][2] for t in range(lo,hi+1) if (e,t) in states];ok=len(vals)==hi-lo+1 and not meta["physical_failure_by_environment"][e]
  windows.append({"ticks":[lo,hi],"target_m":goal,"samples":len(vals),"mean_m":statistics.mean(vals) if ok else None,"mean_absolute_error_m":statistics.mean(abs(v-goal) for v in vals) if ok else None})
 errs=[];expected=0
 for t in range(1,min(1800,meta["steps_by_environment"][e])+1):
  sec=(t-2)*.02
  if 24.5<=sec<=35.0:
   expected+=1;a=states.get((e,t-1));b=targets.get((e,t))
   if a is not None and b is not None:
    delta=max(-.009,min(.006,.03*(final-a["q"][2])-.003));errs.append(abs((b[8]+b[9])/2-.35-(.70+delta)*math.sin(2*math.pi*6.4*(sec-(7.42 if e==6 else 7.46)))))
 valid=all(w["mean_m"] is not None for w in windows);change=windows[1]["mean_m"]-windows[0]["mean_m"] if valid else None
 rows.append({"lane":e,"windows":windows,"height_change_m":change,"late_target_expected_samples":expected,"late_target_samples":len(errs),"late_target_max_error":max(errs,default=None),"passed":valid and all(w["mean_absolute_error_m"]<=.05 for w in windows) and (change>=.12 if e==6 else change<=-.12) and len(errs)==expected and expected>0 and max(errs)<=.0003})
base=json.loads((P/"gate.json").read_text());result={"rows":rows,"base_physical_checks_pass":all(r["passed"] for r in base["rows"]),"passed_switch_screen":all(r["passed"] for r in rows) and all(r["passed"] for r in base["rows"]),"scope":"Separately preregistered within-episode opposite height switches on one used development seed. Not frozen flight, biological validation, takeoff or navigation."};(P/"switch-gate.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
