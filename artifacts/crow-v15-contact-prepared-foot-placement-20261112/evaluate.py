from pathlib import Path
import json,statistics
A=Path("artifacts/crow-v15-contact-prepared-foot-placement-20261112");P=Path("artifacts/crow-v15-knee-phase-advance-screen-20261112");out={}
for seed in (20261126,20261127):
 c=json.loads((A/f"contact-analysis-{seed}.json").read_text());d=json.loads((A/f"displacement-{seed}.json").read_text());j={a:json.loads(((P if a=="baseline" else A)/f"joint-analysis-{seed}-{a}.json").read_text()) for a in ("baseline","phase")};rows=[]
 for e in range(8):
  ratios={}
  for side in ("left","right"):
   z={arm:next(x for x in c["contact_speed"][arm]["rows"] if x["lane"]==e and x["side"]==side)["p95_m_s"] for arm in ("baseline","phase")};assert z["baseline"]>0;ratios[side]=z["phase"]/z["baseline"]
  recovery={}
  for arm in ("baseline","phase"):
   vals=[x for x in j[arm]["records"] if x["lane"]==e];recovery[arm]={"count":len(vals),"left":sum(x["side"]=="left" for x in vals),"right":sum(x["side"]=="right" for x in vals),"knee_median_percent":statistics.median(x["raw"]["knee_minimum_recovery_percent"] for x in vals) if vals else None}
  ph=c["rows"][e]["phase"];progress=d["rows"][e]["candidate"]["dx_m"];base=recovery["baseline"];cand=recovery["phase"];delta=None if base["knee_median_percent"] is None or cand["knee_median_percent"] is None else base["knee_median_percent"]-cand["knee_median_percent"]
  gates={"support":ph["both_unloaded_substeps"]==0,"toe_speed":ph["toe_cap_substeps"]==0,"nonfoot":ph["nonfoot_ground_impulse"]==0,"contact_speed":all(x<=1.5 for x in ratios.values()),"progress":progress>=.05,"recovery_coverage":cand["left"]>0 and cand["right"]>0 and cand["count"]>=base["count"],"knee_phase":delta is not None and delta>=5}
  rows.append({"lane":e,"warp":[0,0,.16,.16,.16,.16,.16,.16][e],"hip_gain":[0,0,.5,.5,.5,.5,.5,.5][e],"ankle_gain":[0,0,-.5,-.5,-.5,-.5,-.5,-.5][e],"gates":gates,"passes":all(gates.values()),"speed_p95_ratio":ratios,"unloaded_substeps":ph["both_unloaded_substeps"],"dx_m":progress,"recovery":recovery,"knee_phase_advance_percent_points":delta,"parity":c["lane_parity"][str(e)]})
 for e in (0,1):assert rows[e]["parity"]=={"endpoints":0,"substeps":0}
 out[str(seed)]=rows
(A/"evaluation.json").write_text(json.dumps(out,indent=2)+"\n")
for seed,rows in out.items():
 for r in rows:print(seed,r["lane"],"pass",r["passes"],"failed",[k for k,v in r["gates"].items() if not v],"p95",{k:round(v,2) for k,v in r["speed_p95_ratio"].items()},"dx",round(r["dx_m"],4),"knee advance",None if r["knee_phase_advance_percent_points"] is None else round(r["knee_phase_advance_percent_points"],1))
