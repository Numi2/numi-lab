import ast,json,math
from pathlib import Path
root=Path(__file__).resolve().parent
prior=root.parent/"mixed-sustained-contact-20261108"
tree=ast.parse((prior/"analyze.py").read_text())
funcs=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
namespace={"DT":.001,"math":math}
exec(compile(funcs,str(prior/"analyze.py"),"exec"),namespace)
analyze=namespace["analyze"]
data={n:json.loads((root/(n+".json")).read_text()) for n in ("control","v2-joint")}
data["fixed128"]=json.loads((prior/"fixed128.json").read_text())
old=json.loads((root.parent/"mixed-contact-sweep-history-20261108/control.json").read_text())
def exposed(x):
 if isinstance(x,dict):return {k:exposed(v) for k,v in x.items() if k not in ("scratch","row_trace")}
 if isinstance(x,list):return [exposed(v) for v in x]
 return x
assert exposed(old)==exposed(data["control"])
for name in ("control","v2-joint"):
 assert all(data[name][k] for k in ("capture_noninvasive","repeat","exact_trace")),name
summaries={n:analyze(d) for n,d in data.items()}
reference=summaries["fixed128"]["steps"]
peak=max(abs(s["ground_normal_impulse_Ns"]/.001) for s in reference)
comparison={}
for name in ("control","v2-joint"):
 rows=summaries[name]["steps"]
 count=sum(s["accepted"] for s in rows)
 assert all(s["accepted"] for s in rows[:count]) and not any(s["accepted"] for s in rows[count:])
 errors=[abs(a["ground_normal_impulse_Ns"]-b["ground_normal_impulse_Ns"])/.001 for a,b in zip(rows[:count],reference)]
 impulse=sum(s["ground_normal_impulse_Ns"] for s in rows[:count])
 refimp=sum(s["ground_normal_impulse_Ns"] for s in reference[:count])
 force=max(errors,default=0.)/peak
 relimp=abs(impulse-refimp)/refimp if refimp else None
 comparison[name]={"accepted_steps":count,"accepted_physical_seconds":count*.001,"first_rejection":count if count<20 else None,
  "full_duration":count==20,"force_error_relative_peak":force,"impulse_relative_error":relimp,
  "force_gate":count==20 and force<=.0025,"impulse_gate":count==20 and relimp is not None and relimp<=.0025,
  "sustained_mixed_steps":summaries[name]["sustained_mixed_steps"],
  "max_normal_residual_m_s":summaries[name]["max_normal_residual_m_s"],
  "max_momentum_balance_Ns":summaries[name]["max_momentum_balance_Ns"],
  "normal_work_estimate_J":summaries[name]["total_ground_normal_work_est_J"],
  "tangent_work_estimate_J":summaries[name]["total_ground_tangent_work_est_J"],
  "rejected_exposed_state_unchanged":all(s["unchanged_from_previous"] for s in data[name]["steps"][count:])}
result={"baseline_reproduced":True,"reference_peak_ground_force_N":peak,"comparison":comparison,
 "per_step":{n:s["steps"] for n,s in summaries.items()}}
(root/"v2-analysis.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="per_step"},indent=2))
