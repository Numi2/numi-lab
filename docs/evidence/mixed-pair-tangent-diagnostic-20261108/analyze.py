import json
import math
from pathlib import Path

root=Path(__file__).resolve().parent
control=json.loads((root/"control.json").read_text())
trace=json.loads((root/"trace.json").read_text())
prior=json.loads((root.parent/"mixed-normal-block-20261108/candidate.json").read_text())
def exposed(value):
    if isinstance(value,dict):
        return {k:exposed(v) for k,v in value.items() if k not in ("scratch","row_trace")}
    if isinstance(value,list): return [exposed(v) for v in value]
    return value
assert exposed(control)==exposed(trace)==exposed(prior), "instrument changed exposed results"
for d in (control,trace):
    assert d["capture_noninvasive"] and d["repeat"] and d["exact_trace"]
    assert d["success"] == 12 and d["failed"] == 8
step=trace["steps"][12]
assert step["code"]==11 and step["unchanged_from_previous"]
out=[]
for i,row in enumerate(step["rows"]):
    z=row["scratch"]
    equation,reported,relative_normal,scale=z[:4]
    rhsu,rhsv,mu,mv=z[4:8]
    velu,velv,regu,regv=z[8:]
    jn,ju,jv=row["impulse"]
    assert all(math.isfinite(v) for v in z+row["impulse"])
    assert scale>0 and jn>=0 and mu>=0 and mv>=0
    lu,lv=mu*jn,mv*jn
    u,v=ju+scale*rhsu,jv+scale*rhsv
    if lu>0 and lv>0:
        n=math.hypot(u/lu,v/lv)
        if n>1: u,v=u/n,v/n
        ratio=math.hypot(ju/lu,jv/lv)
    else:
        u=v=0.
        ratio=None
    computed=math.hypot(u-ju,v-jv)/scale
    assert abs(computed-reported) <= 2e-6, (i,computed,reported)
    out.append({"row":i,"flags":row["flags"],"kind":"rod" if row["flags"] & 8 else "rigid",
      "normal_impulse_Ns":jn,"tangent_impulse_Ns":[ju,jv],"normal_equation_m_s":equation,
      "tangent_rhs_m_s":[rhsu,rhsv],"tangent_relative_velocity_m_s":[velu,velv],
      "tangent_regularization":[regu,regv],"friction":[mu,mv],"cone_utilization":ratio,
      "preconditioner":scale,"reported_tangent_residual_m_s":reported,"cpu_tangent_residual_m_s":computed})
maximum=max(x["cpu_tangent_residual_m_s"] for x in out)
assert abs(maximum-step["failure_tangent_diagnostic"])<=2e-6
dominant=max(out,key=lambda x:x["cpu_tangent_residual_m_s"])
answer={"valid":True,"control_and_trace_match_prior_exposed_state":True,
 "failed_step":12,"status":step["code"],"status_tangent_m_s":step["failure_tangent_diagnostic"],
 "cpu_max_tangent_m_s":maximum,"dominant_row":dominant["row"],"rows":out}
(root/"analysis.json").write_text(json.dumps(answer,indent=2,allow_nan=False)+"\n")
print(json.dumps(answer,indent=2))
