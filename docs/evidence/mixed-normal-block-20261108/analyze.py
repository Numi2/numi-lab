import ast
import json
import math
from pathlib import Path
root = Path(__file__).resolve().parent
prior = root.parent / "mixed-sustained-contact-20261108"
tree = ast.parse((prior / "analyze.py").read_text())
funcs = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
namespace = {"DT": .001, "math": math}
exec(compile(funcs, str(prior / "analyze.py"), "exec"), namespace)
analyze = namespace["analyze"]
data = {n: json.loads((root / (n + ".json")).read_text()) for n in ("baseline", "candidate")}
data["fixed128"] = json.loads((prior / "fixed128.json").read_text())
historical = json.loads((prior / "candidate-2e-5-cap128.json").read_text())
fields = ("world_code", "code", "contacts", "required", "sweeps", "normal_residual", "q", "v", "rod_v", "rows")
mismatches = []
for i, (a, b) in enumerate(zip(data["baseline"]["steps"], historical["steps"])):
    for k in fields:
        if k == "rows":
            for index, (arow, brow) in enumerate(zip(a[k], b[k])):
                if any(arow.get(key) != value for key, value in brow.items()):
                    mismatches.append([i, k, index])
        elif a[k] != b[k]:
            mismatches.append([i, k])
summaries = {n: analyze(d) for n, d in data.items()}
reference = summaries["fixed128"]["steps"]
peak = max(abs(s["ground_normal_impulse_Ns"] / .001) for s in reference)
comparison = {}
for name in ("baseline", "candidate"):
    rows = summaries[name]["steps"]
    count = sum(s["accepted"] for s in rows)
    prefix = rows[:count]
    assert all(s["accepted"] for s in prefix) and not any(s["accepted"] for s in rows[count:])
    errors = [abs(a["ground_normal_impulse_Ns"] - b["ground_normal_impulse_Ns"]) / .001 for a,b in zip(prefix, reference)]
    impulse = sum(s["ground_normal_impulse_Ns"] for s in prefix)
    ref_impulse = sum(s["ground_normal_impulse_Ns"] for s in reference[:count])
    comparison[name] = {
        "accepted_steps": count,
        "accepted_physical_seconds": count * .001,
        "first_rejected_step": count if count < len(rows) else None,
        "max_prefix_1ms_force_error_over_full_reference_peak": max(errors, default=0.) / peak,
        "prefix_ground_impulse_relative_error": abs(impulse-ref_impulse)/ref_impulse if ref_impulse else None,
        "full_duration_gate": "not met: incomplete rollout" if count != len(rows) else "evaluate numerical thresholds",
        "rejected_exposed_q_v_rod_nodes_unchanged": all(s["unchanged_from_previous"] for s in data[name]["steps"][count:]),
        "capture_noninvasive": data[name]["capture_noninvasive"],
        "repeat": data[name]["repeat"],
        "exact_trace": data[name]["exact_trace"],
        "max_accepted_momentum_balance_Ns": summaries[name]["max_momentum_balance_Ns"],
        "accepted_ground_normal_work_est_J": summaries[name]["total_ground_normal_work_est_J"],
        "accepted_ground_tangent_work_est_J": summaries[name]["total_ground_tangent_work_est_J"],
    }
result = {"baseline_historical_mismatches": mismatches, "reference_peak_ground_force_N": peak,
          "comparison": comparison, "per_step": {n:s["steps"] for n,s in summaries.items()}}
(root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k:v for k,v in result.items() if k != "per_step"}, indent=2))
