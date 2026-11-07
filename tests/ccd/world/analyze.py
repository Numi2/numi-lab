#!/usr/bin/env python3
"""Keep full-step collision failures distinct from kernel geometry gates."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("observations",type=Path);a=p.parse_args()
x=json.loads(a.observations.read_text());rows=x["runs"]
assert len(rows)==18 and len({(r["limit"],r["mode"],r["fixture"]) for r in rows})==18
assert all(r["repeat_identical"] and r["rejected_state_restored"] for r in rows)
local=[r for r in rows if r["mode"]==2]
assert all(not r["accepted_unresponded_swept_hit"] for r in local)
assert all(r["accepted"] or r["status"]!=0 for r in rows)
print(json.dumps({"schema":"numi.ccd.complete-analysis.v1",
 "repeated_final_q_v_byte_identical":True,"rejected_q_v_byte_restored":True,
 "interval_local_accepted_steps":sum(r["accepted_steps"] for r in local),
 "interval_local_unresponded_swept_hits":0,
 "accepted_unresponded_swept_hits":[r for r in rows if r["accepted_unresponded_swept_hit"]],
 "decision":"keep disabled; interval-local rejects all tested full steps; legacy fallback misses retained",
 "limits":["Hybrid CCD mode with the production iteration cap of 16; production default CCD mode is speculative.",
 "Zero impulse is a failure here because a force-free principal-axis box sweep analytically crosses the fixed plane.",
 "Positive endpoint gap cannot certify the transient sweep.","Wall times include initialization and are not performance comparisons."]},indent=2))
