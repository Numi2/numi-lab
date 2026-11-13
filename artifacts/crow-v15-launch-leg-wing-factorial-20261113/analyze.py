from pathlib import Path
import collections,gzip,json,sys
A=Path("artifacts/crow-v15-launch-leg-wing-factorial-20261113")
names=("push-wing","nopush-wing","push-nowings","nopush-nowings")
allarms={}
for name in names:
 p=A/f"{name}-20261127-rebound";status=json.loads((p/"run-result.json").read_text());meta=json.loads((p/"run.stdout").read_text())["first_episode_navigation"];s={};c={};caps=collections.Counter();errors=[]
 if status["exit_code"]!=0:errors.append("nonzero exit")
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"]
    if 1<=t<=400:
     assert (e,t) not in s,(name,e,t);s[e,t]=x
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if 1<=t<=400:
     assert (e,t,k) not in c,(name,e,t,k);c[e,t,k]=list(map(float,z[6:11]))
     if int(z[4])>int(z[5]):errors.append(["contact_overflow",e,t,k])
   elif line.startswith("CROW_TOE_V_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if 348<=t<=400 and max(abs(float(v)) for v in z[4:])>=11.99:caps[e]+=1
   elif line.startswith("CROW_SUBSTEP_ERROR"):errors.append(line.strip())
 for e in range(8):
  if meta["steps_by_environment"][e]<400:errors.append(["first_episode_ended",e,meta["steps_by_environment"][e]])
  if sum((e,t) in s for t in range(1,401))!=400:errors.append(["endpoint_missing",e])
  if sum((e,t,k) in c for t in range(1,401) for k in range(16))!=6400:errors.append(["contact_substeps_missing",e])
 allarms[name]={"state":s,"contacts":c,"caps":caps,"errors":errors,"meta":meta}
def same_state(x,y):return all(x[k]==y[k] for k in ("q","v","policy_action","contacts"))
pairs={}
for x,y,stop in (("push-wing","nopush-wing",345),("push-nowings","nopush-nowings",345),("push-wing","push-nowings",315),("nopush-wing","nopush-nowings",315)):
 a,b=allarms[x],allarms[y];rows=[]
 for e in range(8):
  first_s=next((t for t in range(1,401) if not same_state(a["state"][e,t],b["state"][e,t])),None)
  first_c=next(([t,k] for t in range(1,401) for k in range(16) if a["contacts"][e,t,k]!=b["contacts"][e,t,k]),None)
  rows.append({"lane":e,"first_state_difference":first_s,"first_contact_difference":first_c,"prefix_pass":(first_s is None or first_s>stop) and (first_c is None or first_c[0]>stop)})
 pairs[x+" vs "+y]={"prefix_last_step":stop,"rows":rows}
rows={}
for name,a in allarms.items():
 arm=[]
 for e in range(8):
  def impulse(start,end):return [sum(a["contacts"][e,t,k][side] for t in range(start,end+1) for k in range(16)) for side in (0,1)]
  # Aggregate normal impulses at every substep, grouped by control interval.
  loaded=[any(sum(a["contacts"][e,t,k][2:4])>1e-7 for k in range(16)) for t in range(340,401)]
  release=next((340+i for i in range(len(loaded)-1) if not loaded[i] and not loaded[i+1]),None)
  reload=next((340+i for i in range(max((release or 401)-340+2,0),len(loaded)) if loaded[i]),None) if release else None
  states={str(t):{"height_m":a["state"][e,t]["q"][2],"vertical_velocity_m_s":a["state"][e,t]["v"][2],"wing_joint_q":a["state"][e,t]["q"][7:9]} for t in (345,365,380,400)}
  arm.append({"lane":e,"foot_upward_impulse_348_365_ns":impulse(348,365),"foot_upward_impulse_348_400_ns":impulse(348,400),"states":states,"first_two_unloaded_intervals":release,"first_recontact_interval":reload,"toe_speed_cap_substeps":a["caps"][e],"nonfoot_ground_impulse_348_400_ns":sum(a["contacts"][e,t,k][4] for t in range(348,401) for k in range(16))})
 rows[name]=arm
out={"parity":pairs,"arms":rows,"errors":{n:a["errors"] for n,a in allarms.items()},"scope":"Ground impulse includes weight, passive aerodynamics and coupled contact; root velocity is not center-of-mass velocity. Flapping suppression leaves passive aerodynamic forces and other wing actuation. Prefix identity is at recorded endpoints and contact aggregates only. Used seed and correlated lanes, not a biological or held-out mission qualification."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
for name,arm in rows.items():
 for r in arm:print(name,r["lane"],"up_Ns",round(sum(r["foot_upward_impulse_348_365_ns"]),4),"vz365",round(r["states"]["365"]["vertical_velocity_m_s"],4),"z400",round(r["states"]["400"]["height_m"],4),"release",r["first_two_unloaded_intervals"],"reload",r["first_recontact_interval"])
for n,p in pairs.items():print("parity",n,"valid",all(x["prefix_pass"] for x in p["rows"]),"first_state",[x["first_state_difference"] for x in p["rows"]])
print("errors",out["errors"])
if any(out["errors"].values()) or not all(all(x["prefix_pass"] for x in p["rows"]) for p in pairs.values()):raise SystemExit("Invalid arm or pre-intervention parity; do not infer a causal contrast.")
