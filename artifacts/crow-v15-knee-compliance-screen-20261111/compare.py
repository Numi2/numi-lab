from pathlib import Path
import gzip,json
R=Path.cwd();A=R/"artifacts/crow-v15-knee-compliance-screen-20261111"
base={20261126:R/"artifacts/crow-v15-walk-hold-localized-route-20261111/autonomous-20261126",20261127:R/"artifacts/crow-v15-walk-seed27-reference-20261111/autonomous-20261127"}
reports={20261126:R/"artifacts/crow-v15-integrated-walk-window-audit-20261111/autonomous-walk-analysis.json",20261127:R/"artifacts/crow-v15-walk-seed27-reference-20261111/autonomous-walk-analysis.json"}
def knees(path):
 rows={e:[[],[]] for e in range(8)};last=set()
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if not line.startswith("CROW_SUPPORT_SAMPLE "):continue
   z=json.loads(line.split(" ",1)[1]);t=z["step"];e=z["lane"]
   if 201<=t<=300:
    rows[e][0].append(z["q"][15]);rows[e][1].append(z["q"][18])
    if t==300:last.add(e)
   if len(last)==8:break
 assert all(len(v[0])==100 and len(v[1])==100 for v in rows.values())
 return {e:[max(q)-min(q) for q in v] for e,v in rows.items()}
rows=[]
for seed in (20261126,20261127):
 orig=json.loads(reports[seed].read_text())["rows"];new=json.loads((A/f"analysis-{seed}.json").read_text())["rows"];oldk=knees(base[seed]);newk=knees(A/f"autonomous-{seed}-corrected")
 for a,b in zip(orig,new):
  e=a["lane"];ok=oldk[e];nk=newk[e]; om=sum(ok)/2;nm=sum(nk)/2
  rows.append({"seed":seed,"lane":e,"knee_span_baseline_rad":ok,"knee_span_candidate_rad":nk,"bilateral_mean_span_change_fraction":(nm/om-1) if om else None,"unsupported_substeps_baseline":len(a["no_foot_normal_substeps"]),"unsupported_substeps_candidate":len(b["no_foot_normal_substeps"]),"displacement_baseline_m":a["forward_displacement_m"],"displacement_candidate_m":b["forward_displacement_m"],"nonfoot_baseline_intervals":a["nonfoot_contact_intervals"],"nonfoot_candidate_intervals":b["nonfoot_contact_intervals"],"candidate_swing_bouts":[len(b["left_bouts"]),len(b["right_bouts"])]})
result={"rows":rows,"increase_15_percent_lanes":sum(x["bilateral_mean_span_change_fraction"]>=.15 for x in rows),"unsupported_substeps_baseline":sum(x["unsupported_substeps_baseline"] for x in rows),"unsupported_substeps_candidate":sum(x["unsupported_substeps_candidate"] for x in rows),"nonfoot_intervals_baseline":sum(x["nonfoot_baseline_intervals"] for x in rows),"nonfoot_intervals_candidate":sum(x["nonfoot_candidate_intervals"] for x in rows),"positive_displacement_baseline_lanes":sum(x["displacement_baseline_m"]>0 for x in rows),"positive_displacement_candidate_lanes":sum(x["displacement_candidate_m"]>0 for x in rows),"limits":"Angle spans are per knee over endpoint samples; bilateral mean summarizes each lane. The perturbation is active before walking, so trajectories can diverge before the walk. Ground and reconstructed geometry validation are in the individual analysis files. No landing, takeoff or biological conclusion."}
(A/"comparison.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:v for k,v in result.items() if k not in ("rows","limits")},indent=2))
