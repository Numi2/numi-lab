from pathlib import Path
import collections,gzip,json,math,types
A=Path("artifacts/crow-v15-six-second-walk-screen-20261112");P=Path("artifacts/crow-v15-matched-binary-gait-screen-20261112")
def load(path):
 result=json.loads((path/"run.stdout").read_text())["first_episode_navigation"];states={};sub={};caps=collections.Counter();vel=collections.Counter();errors=[]
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if "CROW_SUBSTEP_ERROR" in line:errors.append(line.strip())
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"]
    if 1<=t<=500:
     assert (e,t) not in states;states[e,t]=x
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=500:
     assert (e,t,k) not in sub;sub[e,t,k]=list(map(float,z[6:]))
     if int(z[4])>int(z[5]):errors.append("overflow")
   elif line.startswith("CROW_TOE_V_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 201<=t<=500:
     vel[e,t,k]+=1
     if max(abs(float(v)) for v in z[4:])>=11.99:caps[e]+=1
 return result,states,sub,caps,errors,vel
arms={"sham":load(A/"sham-20261126-dev"),"candidate":load(A/"candidate-20261126-rebound")};old={"sham":load(P/"sham-20261126"),"candidate":load(P/"candidate-20261126-rebound")}
rows=[]
for e in range(8):
 s,c=arms["sham"],arms["candidate"];prefix=[];contact=[];old_mismatch={}
 for t in range(1,200):
  if any(s[1].get((e,t),{}).get(k)!=c[1].get((e,t),{}).get(k) for k in ("q","v","policy_action","contacts")):prefix.append(t)
  for k in range(16):
   if s[2].get((e,t,k))!=c[2].get((e,t,k)):contact.append([t,k])
 for name in arms:
  old_mismatch[name]=next((t for t in range(1,301) if any(arms[name][1].get((e,t),{}).get(k)!=old[name][1].get((e,t),{}).get(k) for k in ("q","v","policy_action","contacts"))),None)
 by={}
 for name,(r,st,sub,caps,err,vel) in arms.items():
  end=r["steps_by_environment"][e];missing=sum((e,t,k) not in sub for t in range(1,min(500,end)+1) for k in range(16));unloaded=sum(sub[e,t,k][2]<=1e-7 and sub[e,t,k][3]<=1e-7 for t in range(201,min(500,end)+1) for k in range(16) if (e,t,k) in sub);nonfoot=sum(abs(sub[e,t,k][4])>1e-7 for t in range(201,min(500,end)+1) for k in range(16) if (e,t,k) in sub);fall=next((t for t in range(201,min(500,end)+1) if (e,t) in st and st[e,t]["q"][2]<.1),None)
  by[name]={"end":end,"physical_failure":r["physical_failure_by_environment"][e],"reason":r["termination_reason_by_environment"][e],"missing_contact_substeps":missing,"missing_velocity_substeps":sum(vel[e,t,k]!=1 for t in range(201,min(500,end)+1) for k in range(16)),"both_unloaded_substeps":unloaded,"nonfoot_substeps":nonfoot,"toe_cap_substeps":caps[e],"first_height_below_0_1m":fall}
 rows.append({"lane":e,"first_prefix_state_mismatch":prefix[:1],"first_prefix_contact_mismatch":contact[:1],"first_mismatch_to_two_second_build":old_mismatch,"arms":by})
coverage=all(arms[n][0]["steps_by_environment"][e]>=500 and not arms[n][0]["physical_failure_by_environment"][e] and all((e,t) in arms[n][1] for t in range(201,501)) for n in arms for e in range(8))
windows=[]
if coverage:
 source=Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.py").read_text();oldexpr="if 200<=v[\x27step\x27]<=299";assert source.count(oldexpr)==1;source=source.replace(oldexpr,"if LO-1<=v[\x27step\x27]<=HI-1");mod=types.ModuleType("audit_window");exec(compile(source,"audit_window","exec"),mod.__dict__)
 for lo in (201,301,401):
  mod.LO=lo;mod.HI=lo+99;w={"first":lo,"last":lo+99,"arms":{}}
  for n,tag in (("sham","dev"),("candidate","rebound")):
   w["arms"][n]=mod.analyze(A/(n+"-20261126-"+tag),0.0)
  windows.append(w)
 for row in rows:
  e=row["lane"];row["window_gates"]=[{"first":w["first"],"candidate_travel_m":w["arms"]["candidate"][e]["forward_root_displacement_m"],"sham_travel_m":w["arms"]["sham"][e]["forward_root_displacement_m"],"candidate_swing_bouts":[len(v) for v in w["arms"]["candidate"][e]["support_qualified_swing_bouts"]]} for w in windows]
  row["pass"]=not row["first_prefix_state_mismatch"] and not row["first_prefix_contact_mismatch"] and all(not row["arms"][n][k] for n in arms for k in ("physical_failure","missing_contact_substeps","missing_velocity_substeps","nonfoot_substeps","toe_cap_substeps","first_height_below_0_1m")) and all(g["candidate_travel_m"]>0 and min(g["candidate_swing_bouts"])>=2 for g in row["window_gates"])
out={"coverage_complete":coverage,"rows":rows,"errors":{n:arms[n][4] for n in arms},"windows":windows,"all_gates_pass":coverage and all(r.get("pass",False) for r in rows) and all(not arms[n][4] for n in arms),"scope":"Six-second matched development walk; endpoint clearance only, no biological or full-sequence qualification."};(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k!="windows"},indent=2))
