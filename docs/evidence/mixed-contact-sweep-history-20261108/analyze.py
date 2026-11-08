import json
import math
from pathlib import Path
root=Path(__file__).resolve().parent
prior=json.loads((root.parent/"mixed-normal-block-20261108/candidate.json").read_text())
prior_trace=json.loads((root.parent/"mixed-pair-tangent-diagnostic-20261108/trace.json").read_text())
control=json.loads((root/"control.json").read_text())
def exposed(value):
    if isinstance(value,dict):
        return {k:exposed(v) for k,v in value.items() if k not in ("scratch","row_trace")}
    if isinstance(value,list): return [exposed(v) for v in value]
    return value
assert exposed(control)==exposed(prior)
assert control["capture_noninvasive"] and control["repeat"] and control["exact_trace"]
assert control["success"]==12 and control["failed"]==8
series=[]
for i in (1,32,96,120,124,125,126,127,128):
    trial=json.loads((root/("sweep-%03d.json"%i)).read_text())
    assert exposed(trial)==exposed(control), i
    assert all(trial[k] for k in ("capture_noninvasive","repeat","exact_trace")), i
    step=trial["steps"][12]
    assert step["code"]==11 and step["unchanged_from_previous"] and len(step["rows"])==3
    rows=[]
    for k,row in enumerate(step["rows"]):
        z=row["scratch"]
        assert len(z)==12 and all(math.isfinite(v) for v in z)
        jn,ju,jv,marker,rn,ru,rv,scale,vn,vu,vv,relax=z
        assert marker==i and jn>=0 and scale>0
        limit=.5*jn
        u=ju+scale*ru; v=jv+scale*rv
        if limit>0:
            ratio=math.hypot(u/limit,v/limit)
            if ratio>1: u/=ratio; v/=ratio
            utilization=math.hypot(ju/limit,jv/limit)
        else: u=v=0.;utilization=None
        tangent=math.hypot(u-ju,v-jv)/scale
        rows.append({"row":k,"rod":bool(row["flags"]&8),"impulse":z[:3],
            "normal_rhs":rn,"tangent_rhs":[ru,rv],"scale":scale,
            "relative_row_velocity":[vn,vu,vv],"relaxation":relax,
            "friction_utilization":utilization,"tangent_residual":tangent})
        if i==128:
            old=prior_trace["steps"][12]["rows"][k]
            assert all(abs(a-b)<2e-9 for a,b in zip(z[:3],old["impulse"]))
            assert abs(tangent-old["scratch"][1])<2e-6
            assert abs(ru-old["scratch"][4])<2e-6 and abs(rv-old["scratch"][5])<2e-6
            assert abs(scale-old["scratch"][3])<2e-6
    series.append({"iteration":i,"rows":rows})
out={"non_scratch_traces_match_control_and_prior":True,"control":"12 accepted, 8 rejected; step 12 status 11",
     "friction_coefficient":.5,"final_capture_agrees_with_prior":True,"series":series}
(root/"analysis.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
for entry in series:
    print("iteration",entry["iteration"],"rows",[(r["row"],round(r["tangent_residual"],8),[round(x,10) for x in r["impulse"]],round(r["friction_utilization"],5) if r["friction_utilization"] is not None else None) for r in entry["rows"]])
