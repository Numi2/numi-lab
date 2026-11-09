#!/usr/bin/env python3
"""Validate recorded FP64 contact-work histories and temporal refinement."""
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
receipt = json.loads((e / "slider-work-history-reference-002-execution.json").read_text())
if receipt["result"] != "four references completed with exact historical-field parity" or len(receipt["runs"]) != 4:
    raise ValueError("incomplete reference receipt")

def read(fixture, label, run_index):
    path = e / ("slider-work-history-002-" + fixture + "-h" + label + ".json")
    raw = path.read_bytes()
    run = receipt["runs"][run_index]
    if (not run["historic_fields_exact_match"] or run["output_sha256"] != hashlib.sha256(raw).hexdigest()):
        raise ValueError("reference output does not match its execution record")
    x = json.loads(raw)
    history = x["work_trace"]
    if len(history) != 5001 or len(x["trace"]) != 5001:
        raise ValueError("incomplete work history")
    worst = 0.
    for i, (row, other) in enumerate(zip(history, x["trace"])):
        if (len(row) != 4 or not all(math.isfinite(v) for v in row) or
            row[0] != other[0] or abs(row[0] - i * 2e-6) > 1e-12):
            raise ValueError("invalid work-history row or clock")
        scale = 4096 * sys.float_info.epsilon * sum(abs(v) for v in row[1:]) + 1e-20
        worst = max(worst, abs(row[1] + row[2] - row[3]) / scale)
    if worst > 1 or history[0][1:] != [0, 0, 0] or history[-1][1:] != [x["fem_contact_work_J"], x["slider_contact_work_J"], x["relative_contact_work_J"]]:
        raise ValueError("inconsistent cumulative work components or endpoints")
    return x, worst

result = {"schema": "numi.slider-work-reference-refinement.v1", "fixtures": {},
          "limits": {"history_relative_to_max_or_floor": .01, "work_absolute_floor_J": 1e-12},
          "scope": "Independent FP64 reference histories only; no native transaction, material calibration, or rollout performance result."}
for fixture, offset in (("primary", 0), ("near", 2)):
    coarse, balance_a = read(fixture, "050", offset)
    fine, balance_b = read(fixture, "025", offset + 1)
    rows = {}
    for col, name in enumerate(("fem", "slider", "relative"), 1):
        a, b = coarse["work_trace"], fine["work_trace"]
        peak = max(abs(row[col]) for row in b)
        maximum = max(abs(u[col] - v[col]) for u, v in zip(a, b))
        final = abs(a[-1][col] - b[-1][col])
        ratio = maximum / max(peak, 1e-12)
        rows[name] = {"fine_maximum_absolute_cumulative_J": peak,
                      "maximum_coarse_fine_absolute_J": maximum,
                      "final_coarse_fine_absolute_J": final,
                      "maximum_normalized_error": ratio, "passes": ratio <= .01}
    result["fixtures"][fixture] = {"histories": rows,
        "largest_work_component_balance_ratio": max(balance_a, balance_b),
        "passes": all(x["passes"] for x in rows.values())}
result["passes"] = all(x["passes"] for x in result["fixtures"].values())
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["passes"] else 1)
