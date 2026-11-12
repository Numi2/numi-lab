from pathlib import Path
import gzip,json,math,collections
p=Path("artifacts/crow-v15-rear-toe-attachment-contact-20261112/candidate-smoke-20261127")
a=json.loads((p/"run.stdout").read_text()); command=json.loads((p/"command.json").read_text()); samples=collections.Counter(); errors=[]
for k in ("failed_environment_steps","height_or_tilt_termination_count","host_requested_resets","termination_count"):
 if a[k]!=0: errors.append([k,a[k]])
if any(a["physics_error_count_by_environment"]):errors.append(["physics_error_count_by_environment",a["physics_error_count_by_environment"]])
if command[command.index("--steps")+1]!="2": errors.append(["steps",command])
with gzip.open(p/"run.stderr.gz","rt") as f:
 for line in f:
  prefix,_,data=line.partition(" ")
  if prefix in ("CROW_FLIGHT_SAMPLE","CROW_SUPPORT_SAMPLE"):
   obj=json.loads(data);samples[prefix]+=1
   for k in ("q","v","policy_action"):
    if k in obj and not all(math.isfinite(x) for x in obj[k]):errors.append([prefix,obj.get("step"),obj.get("lane"),k])
  elif prefix in ("CROW_TWO_TOE_SUBSTEP","CROW_TOE_V_SUBSTEP","CROW_PHASE_SAMPLE"):
   samples[prefix]+=1
   if not all(math.isfinite(float(x)) for x in data.split()): errors.append([prefix,"nonfinite"])
if samples != {"CROW_FLIGHT_SAMPLE":24,"CROW_SUPPORT_SAMPLE":24,"CROW_PHASE_SAMPLE":24,"CROW_TWO_TOE_SUBSTEP":384,"CROW_TOE_V_SUBSTEP":384}: errors.append(["sample_counts",dict(samples)])
res={"steps_requested":2,"environments":8,"sample_counts":dict(samples),"maximum_active_contacts":a["maximum_active_contacts"],"errors":errors,"passed_short_smoke":not errors,"caveat":"This two-step diagnostic precedes the longer changed-body trial and does not establish walking, takeoff, flight, or landing qualification."}
(p/"smoke-check.json").write_text(json.dumps(res,indent=2)+"\n");print(json.dumps(res,indent=2));assert not errors
