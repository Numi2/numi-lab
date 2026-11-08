import json, math
from pathlib import Path
p=Path(__file__).resolve().parent
out={}
for arm in ("warm","cold"):
    assert (p/(arm+".exit")).read_text().strip()=="0", arm+" failed"
    data=json.loads((p/(arm+".json")).read_text())
    assert len(data["cases"])==5
    cases=[]
    for i, case in enumerate(data["cases"]):
        steps=case["steps"]
        assert len(steps)==100
        for s in steps:
            assert all(math.isfinite(x) for x in s["q"]+s["v"]+[s["linear_error"],s["angular_error"],s["normal_impulse"],s["residual"],s["energy"]])
        accepted=[x for x in steps if x["accepted"]]
        rejected=[j+1 for j,x in enumerate(steps) if not x["accepted"]]
        item={"case":i,"accepted":len(accepted),"first_rejected":rejected[0] if rejected else None,
              "replay":all(x["replay"] for x in steps),"restored":all(x["restored"] for x in steps if not x["accepted"]),
              "max_linear_error":max((x["linear_error"] for x in accepted),default=0),
              "max_angular_error":max((x["angular_error"] for x in accepted),default=0),
              "max_normal_impulse":max((abs(x["normal_impulse"]) for x in accepted),default=0),
              "max_normal_residual":max((abs(x["residual"]) for x in accepted),default=0)}
        if i<2:
            item["second_impulse"]=steps[1]["normal_impulse"]
            item["second_residual"]=steps[1]["residual"]
            item["gate"]=len(accepted)==100 and item["replay"] and abs(item["second_impulse"])<=.01 and abs(item["second_residual"])<=.002
        elif i==2:
            item["energy_error"]=max(abs(x["energy"]-.5) for x in steps)
            item["gate"]=len(accepted)==100 and item["replay"] and item["max_normal_impulse"]==0 and item["energy_error"]<1e-6 and item["max_linear_error"]<2e-6
        elif i==3:
            item["static_impulse_error"]=max(abs(x["normal_impulse"]-.0981) for x in steps)
            item["gate"]=len(accepted)==100 and item["replay"] and item["static_impulse_error"]<.0002 and item["max_linear_error"]<2e-6
        else:
            item["gate"]=len(accepted)==100 and item["replay"] and item["max_linear_error"]<2e-6
        cases.append(item)
    out[arm]=cases
out["all_gates_pass"]=all(c["gate"] for arm in ("warm","cold") for c in out[arm])
(p/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
