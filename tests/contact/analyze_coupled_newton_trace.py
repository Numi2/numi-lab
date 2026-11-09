#!/usr/bin/env python3
"""Summarize the observational trace; no stopping criterion is changed."""
import json
from pathlib import Path
import sys
p=Path(sys.argv[1])
rows=[]
for line in p.read_text().splitlines():
    if line.startswith("fem_newton_iterate="):
        r=json.loads(line.split("=",1)[1])
        if r["stage"] == "final_line_search":
            rows.append({"iteration":r["iteration"], "alpha":r["alpha"][0],
                         "reassembled_nonlinear_norm":r["fgmres"][0]["nonlinear"][1],
                         "reported_linear_metric":r["fgmres"][0]["diagnostics"][0],
                         "reported_cycle_accepted":r["fgmres"][0]["cycle_accepted"],
                         "object_limit":r["object_line_search"][0][0],
                         "status":r["status"][0]["code"]})
if len(rows)!=16 or [x["iteration"] for x in rows]!=list(range(16)):
    raise ValueError("expected all 16 iterations of the selected failure")
print(json.dumps({"rows":rows,"all_selected_alpha_one":all(x["alpha"]==1 for x in rows),
 "all_object_limits_one":all(x["object_limit"]==1 for x in rows),
 "last_assembled_nonlinear_norm":rows[-1]["reassembled_nonlinear_norm"],
 "limitations":["solver metrics combine mixed coordinate units",
 "linear metric is runtime-reported, not independently recomputed",
 "line-search samples do not certify the full Armijo decision process",
 "this is legacy mixed FEM with a reacting body, not the reference-area midpoint fixture"]},indent=2))
