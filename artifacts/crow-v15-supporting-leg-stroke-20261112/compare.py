from pathlib import Path
import collections,gzip,importlib.util,json,math
A=Path("artifacts/crow-v15-supporting-leg-stroke-20261112")
paths={"sham":Path("artifacts/crow-v15-upper-body-transition-isolation-20261112/sham-20261127-held"),"candidate":A/"sham-20261127-stance"}
spec=importlib.util.spec_from_file_location("audit",str(A/"long-audit.py"));audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
old=json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text());ground=next(x["ground_z_m"] for x in old["rows"] if x["seed"]==20261127)
def parse(path):
 states={};sub={};errors=[];caps=collections.Counter();coverage=collections.Counter();ep=json.loads((path/"run.stdout").read_text())["first_episode_navigation"]
 assert json.loads((path/"run-result.json").read_text())["exit_code"]==0
 assert ep["steps_by_environment"]==[500]*8 and not any(ep["closed_by_environment"]) and not any(ep["physical_failure_by_environment"]),ep
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if "CROW_SUBSTEP_ERROR" in line:errors.append(line.strip())
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    z=json.loads(line.split(" ",1)[1]);key=z["lane"],z["step"]
    if 1<=key[1]<=500:
     assert key not in states;states[key]=[z[k] for k in ("q","v","policy_action","contacts")]
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);key=e,t,k
    if 1<=t<=500:
     assert key not in sub
     if int(z[4])>int(z[5]):errors.append("constraint overflow")
     sub[key]=list(map(float,z[6:]))
   elif line.startswith("CROW_TOE_V_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if 201<=t<=500:
     coverage[e]+=1
     if max(abs(float(x)) for x in z[4:])>=11.99:caps[e]+=1
 assert len(states)==4000 and len(sub)==64000,(len(states),len(sub))
 return states,sub,caps,coverage,errors
arms={k:parse(p) for k,p in paths.items()};gait={k:audit.analyze(p,ground) for k,p in paths.items()};c,r=arms["candidate"],arms["sham"]
rows=[]
for e in range(8):
 prefix=sum(c[0][e,t]!=r[0][e,t] for t in range(1,201));prefixc=sum(c[1][e,t,k]!=r[1][e,t,k] for t in range(1,201) for k in range(16))
 def un(a):return sum(a[1][e,t,k][2]<=1e-7 and a[1][e,t,k][3]<=1e-7 for t in range(201,501) for k in range(16))
 def bad(a):return [t for t in range(201,501) if any(a[1][e,t,k][4]>1e-7 for k in range(16))]
 candidate=gait["candidate"][e];sham=gait["sham"][e];ct=candidate["forward_root_displacement_m"];rt=sham["forward_root_displacement_m"]
 row={"lane":e,"prefix_state_mismatches":prefix,"prefix_substep_mismatches":prefixc,"unloaded_substeps_candidate":un(c),"unloaded_substeps_sham":un(r),"nonfoot_intervals_candidate":bad(c),"nonfoot_intervals_sham":bad(r),"toe_velocity_cap_samples_candidate":c[2][e],"toe_velocity_cap_coverage_candidate":c[3][e],"candidate":candidate,"sham":sham}
 row["pass"]=not prefix and not prefixc and un(c)<=un(r) and not bad(c) and not bad(r) and min(map(len,candidate["support_qualified_swing_bouts"]))>=10 and ct>0 and rt>0 and ct>=.9*rt and not c[2][e]
 rows.append(row)
drag={n:sum(sum(row["sole_clear_but_toes_below_clearance_intervals"]) for row in g) for n,g in gait.items()}
out={"seed":20261127,"window_control_steps":[201,500],"lane_rows":rows,"sole_clear_toe_low":drag,"errors":{n:x[4] for n,x in arms.items()},"all_declared_gates_pass":all(x["pass"] for x in rows) and sum(z["candidate"]["forward_root_displacement_m"] >= 1.5*z["sham"]["forward_root_displacement_m"] for z in rows)>=6 and all(not x[4] for x in arms.values()),"scope":"One previously used seed; six seconds; physical control and simulated contacts only. Toe velocity caps assessed only at recorded substeps; consult coverage. Sole-clear/toe-low endpoints are descriptive here. No biological or takeoff/landing/flight/navigation claim."}
(A/"comparison-20261127.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"pass":out["all_declared_gates_pass"],"toe_low":drag,"rows":[{"lane":z["lane"],"pass":z["pass"],"travel_fraction":z["candidate"]["forward_root_displacement_m"]/z["sham"]["forward_root_displacement_m"] if z["sham"]["forward_root_displacement_m"] else None,"bouts":[len(q) for q in z["candidate"]["support_qualified_swing_bouts"]],"unloaded":[z["unloaded_substeps_candidate"],z["unloaded_substeps_sham"]],"toe_speed_caps":z["toe_velocity_cap_samples_candidate"],"toe_speed_coverage":z["toe_velocity_cap_coverage_candidate"]} for z in rows]},indent=2))
