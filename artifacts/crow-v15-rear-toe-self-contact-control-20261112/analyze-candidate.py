from pathlib import Path
import importlib.util,json,collections,gzip
p=Path("artifacts/crow-v15-rear-toe-self-contact-control-20261112");s=importlib.util.spec_from_file_location("audit",p/"audit-ratio.py");a=importlib.util.module_from_spec(s);s.loader.exec_module(a);ground=next(x["ground_z_m"] for x in json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text())["rows"] if x["seed"]==20261127);d=p/"candidate-20261127";episode=json.loads((d/"run.stdout").read_text())["first_episode_navigation"];assert episode["steps_by_environment"]==[300]*8 and not any(episode["physical_failure_by_environment"]);rows=a.analyze(d,ground);sub={}
with gzip.open(d/"run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=l.split();t,k,e=map(int,z[1:4]);
   if 201<=t<=300:sub[e,t,k]=list(map(float,z[6:]))
assert len(sub)==12800
for e,r in enumerate(rows):r["both_zero_substeps"]=sum(sub[e,t,k][2]<=1e-7 and sub[e,t,k][3]<=1e-7 for t in range(201,301) for k in range(16));r["nonfoot_intervals"]=[t for t in range(201,301) if sum(sub[e,t,k][4] for k in range(16))>1e-7]
out={"rows":rows,"low_toe_count":sum(sum(r["sole_clear_but_toes_below_clearance_intervals"]) for r in rows),"scope":"Collision-exclusion diagnostic only. Candidate specific 40-DOF FK and offset sole shape; validates against lane-5 replay. A clean diagnostic sequence is not a physical qualification."};(p/"candidate-gait.json").write_text(json.dumps(out,indent=2)+"\n")
for r in rows:print(r["lane"],"travel",r["forward_root_displacement_m"],"bouts",[len(x) for x in r["support_qualified_swing_bouts"]],"zero",r["both_zero_substeps"],"nonfoot",r["nonfoot_intervals"])
print("low toe",out["low_toe_count"])
