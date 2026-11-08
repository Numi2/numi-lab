import json, math
from pathlib import Path
p=Path(__file__).resolve().parent
out={}
for arm in ("warm","cold"):
    baseline=json.loads((p/f"{arm}.json").read_text())
    detail=json.loads((p/f"friction-{arm}.json").read_text())
    check=json.loads((p/f"friction-{arm}-check.json").read_text())
    assert len(baseline["cases"])==len(detail["cases"])==5
    assert check["failure_count"]==0 and not check["failures"]
    row_count=sliding=0
    for basic,case in zip(baseline["cases"],detail["cases"]):
        assert len(basic["steps"])==len(case["steps"])==100
        for i,(x,s) in enumerate(zip(basic["steps"],case["steps"])):
            assert s["accepted"] and s["replay"]
            assert x["q"]==s["q"] and x["v"]==s["v"] and x["normal_impulse"]==s["normal_impulse"]
            q=case["steps"][i-1]["q"] if i else [0,.01+case["initial_gap"],0]
            v=s["v"]
            for row in s["rows"]:
                vals=row["point"]+row["directions"][0]+row["directions"][1]+row["directions"][2]+row["tangent_targets"]+row["tangent_impulses"]+[row["impulse"]]
                assert all(math.isfinite(z) for z in vals)
                assert row["impulse"]>=0
                r=[row["point"][j]-q[j] for j in range(3)];w=v[3:]
                velocity=[v[0]+w[1]*r[2]-w[2]*r[1],v[1]+w[2]*r[0]-w[0]*r[2],v[2]+w[0]*r[1]-w[1]*r[0]]
                vt=[sum(a*b for a,b in zip(row["directions"][k],velocity))-row["tangent_targets"][k-1] for k in (1,2)]
                row_count+=1; sliding+=math.hypot(*vt)>1e-5
    out[arm]={"accepted_replayed_steps":500,"checked_rows":row_count,"sliding_rows":sliding,"reported_failures":check["failure_count"],"state_and_normal_impulse_match_prior_run":True}
(p/"friction-verification.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
