from pathlib import Path
import collections,gzip,json,math,sys
A=Path("artifacts/crow-v15-mirrored-leg-wing-longflight-20261113");P=A/sys.argv[1];seed=next(v for v in ("20261126","20261127") if v in P.name);R=Path("artifacts/crow-v15-mirrored-amplitude-screen-20261113")/("pronation-small-20261126-rebound-small" if seed=="20261126" else "pronation-small-20261127-rebound-repeat")
def read(p):
 out={"states":{},"contacts":collections.defaultdict(lambda:[0.,0.,0]),"toe":collections.Counter(),"toe_cap":collections.Counter(),"effort":collections.Counter(),"effort_max":collections.defaultdict(float),"target":{},"ground":{},"errors":[],"mask":[]}
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);key=x["lane"],x["step"]
    if key in out["states"]:out["errors"].append(["duplicate state",key])
    out["states"][key]=x;continue
   z=l.split()
   if not z:continue
   kind=z[0]
   if kind=="CROW_SCENE_MASK":out["mask"].append(z[1:]);continue
   if "OBSERVER_ERROR" in kind or kind=="CROW_SUBSTEP_ERROR":out["errors"].append(l.strip());continue
   if kind=="CROW_DRIVE_TARGET":
    t,e=map(int,z[1:3]);out["target"][e,t]=list(map(float,z[3:]));continue
   if kind in ("CROW_TWO_TOE_SUBSTEP","CROW_TOE_V_SUBSTEP","CROW_WING_EFFORT","CROW_GROUND_CONTACT"):
    t,k,e=map(int,z[1:4]);key=e,t
    if kind=="CROW_TWO_TOE_SUBSTEP":
     a=out["contacts"][key];a[0]+=sum(map(float,z[8:10]));a[1]+=float(z[10]);a[2]+=1
    elif kind=="CROW_TOE_V_SUBSTEP":
     v=list(map(float,z[4:]));out["toe"][key]+=1
     if len(v)!=16 or not all(map(math.isfinite,v)):out["errors"].append(["invalid toe",key])
     elif max(map(abs,v))>=11.99:out["toe_cap"][key]+=1
    elif kind=="CROW_WING_EFFORT":
     v=list(map(float,z[4:]));out["effort"][key]+=1
     if len(v)!=14 or not all(map(math.isfinite,v)):out["errors"].append(["invalid effort",key])
     else:out["effort_max"][e]=max(out["effort_max"][e],abs(v[0]),abs(v[7]))
    else:out["ground"].setdefault(key,l.strip())
 return out
new,ref=read(P),read(R);m=json.loads((P/"run.stdout").read_text());meta=m["first_episode_navigation"];ni=json.loads((P/"inputs.json").read_text());ri=json.loads((R/"inputs.json").read_text());source=(A/"signed-turn.metal").read_text();shared=all(ni[k]==ri[k] for k in ("host_sha256","landing_policy_sha256","route_policy_sha256")) and new["mask"]==ref["mask"] and not new["errors"] and "if (taskStates[environment].episode.z == 10u) requested = 0.0f;" in source and m["birdflow_journey_teacher"] is False
parity=[]
for e in range(8):
 refmeta=json.loads((R/"run.stdout").read_text())["first_episode_navigation"];stop=refmeta["steps_by_environment"][e] if e<6 else 310;bad=[]
 for t in range(1,stop+1):
  a=new["states"].get((e,t));b=ref["states"].get((e,t))
  if a is None or b is None or any(a[k]!=b[k] for k in ("q","v","policy_action","contacts")):bad.append(t)
 parity.append({"lane":e,"stop":stop,"mismatches":bad})
rows=[]
for e in (6,7):
 end=min(meta["steps_by_environment"][e],1800);release=next((t for t in range(371,end-1) if all(new["contacts"][e,u][2]==16 and new["contacts"][e,u][0]<=1e-7 for u in range(t,t+3))),None)
 s=new["states"];firstground=next(((t,new["ground"][e,t]) for t in range((release or end)+1,end+1) if (e,t) in new["ground"]),None);recontact=next((t for t in range(release+3,end+1) if new["contacts"][e,t][0]>1e-7),None) if release is not None else None
 states_ok=all((e,t) in s for t in range(1,end+1));coverage=all(new["contacts"][e,t][2]==16 and new["toe"][e,t]==16 and new["effort"][e,t]==16 and (e,t) in new["target"] for t in range(348,end+1));delta=s[e,1800]["q"][0]-s[e,release]["q"][0] if end==1800 and release is not None and (e,1800) in s and (e,release) in s else None;heights=[s[e,t]["q"][2] for t in range(450,end+1) if (e,t) in s];vxs=[s[e,t]["v"][0] for t in range(550,end+1) if (e,t) in s];errs=[];start=7.42 if e==6 else 7.46
 for t in range(390,end+1):
  v=new["target"].get((e,t));sec=(t-2)*.02
  if v is not None:a=s.get((e,t-1)); u=max(0.,min(1.,(sec-12.)/.5)); envelope=u*u*(3.-2.*u); amplitude_delta=max(-.006,min(.006,.03*((.80 if e==6 else .55)-a["q"][2])-.003))*envelope if a is not None else float("inf"); errs.append(abs((v[8]+v[9])/2-.35-(.70+amplitude_delta)*math.sin(2*math.pi*6.4*(sec-start))))
 checks={"shared":shared and all(not p["mismatches"] for p in parity),"first_episode":end==1800 and not meta["physical_failure_by_environment"][e],"coverage":states_ok and coverage,"release":release is not None,"no_ground_after_release":release is not None and recontact is None and firstground is None and sum(new["contacts"][e,t][1] for t in range(release,end+1))<=1e-7,"no_toe_cap":sum(new["toe_cap"][e,t] for t in range(348,end+1))==0,"actuator_bounds":new["effort_max"][e]<=3.00001,"wing_target":len(errs)==end-389 and bool(errs) and max(errs)<=.0003,"forward_progress":delta is not None and delta>=.10 and len(vxs)==1251 and min(vxs)>0 and max(vxs)<=1.,"height":len(heights)==1351 and min(heights)>=.25 and max(heights)<=1.5};rows.append({"lane":e,"checks":checks,"release":release,"first_recontact":recontact,"first_recorded_ground":firstground,"root_x_displacement_release_to_1800_m":delta,"root_vx_550_1800_range":[min(vxs),max(vxs)] if vxs else None,"height_range_450_1800":[min(heights),max(heights)] if heights else None,"toe_cap_count":sum(new["toe_cap"][e,t] for t in range(348,end+1)),"wing_target_error_max":max(errs,default=None),"passed":all(checks.values())})
out={"parity":parity,"rows":rows,"passed_any":any(r["passed"] for r in rows),"scope":"One used-seed 1800-step first-episode flight screen; no qualification of sustained flight, takeoff, landing, navigation, or biology."};(P/"gate.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
