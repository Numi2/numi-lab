#!/usr/bin/env python3
"""Keep full-step collision failures distinct from kernel geometry gates."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("observations",type=Path);a=p.parse_args()
x=json.loads(a.observations.read_text());rows=x["runs"]
assert len(rows)==288 and len({(r["scale"],r["limit"],r["mode"],r["fixture"]) for r in rows})==288
assert all(r["repeat_identical"] and r["rejected_state_restored"] for r in rows)
local=[r for r in rows if r["mode"]==2]
assert all(not r["accepted_unresponded_swept_hit"] for r in rows)
# The two thinner impact stresses still exhaust certified progress/events.
# Keep that failed all-impact gate explicit; the regression separately requires
# useful original/unused-rotation impacts and all production-cap clear controls.
full_resolution=all(r["accepted"] for r in rows if r["limit"]>=16)
assert all(r["accepted"] for r in rows if r["limit"]>=16 and r["fixture"] not in (12,14))
assert all(r["accepted"] or r["status"] in (9,11) for r in rows if r["fixture"] in (12,14))
assert all(r["normal_impulse_Ns"]>0 for r in rows if r["accepted"] and r["free_sweep_min_gap"]<0)
assert all(r["normal_impulse_Ns"]==0 for r in rows if r["accepted"] and r["free_sweep_min_gap"]>0)
assert all(r["rejected_contact_cache_empty"] for r in rows)
assert any(not r["accepted"] for r in rows if r["limit"]==1)
assert all(r["accepted"] or r["status"]!=0 for r in rows)
assert all(not r["accepted"] for r in rows if r["unresolved_ccd"]>0)
print(json.dumps({"schema":"numi.ccd.complete-analysis.v1",
 "repeated_final_q_v_byte_identical":True,"rejected_q_v_byte_restored":True,
 "interval_local_accepted_steps":sum(r["accepted_steps"] for r in local),
 "interval_local_unresponded_swept_hits":0,
 "accepted_unresponded_swept_hits":[r for r in rows if r["accepted_unresponded_swept_hit"]],
 "production_cap_original_and_unused_rotation_impacts_accept":True,
 "production_cap_clear_controls_accept":True,
 "all_impact_resolution_gate_passed":full_resolution,
 "remaining_rejected_production_cap_cases":[{"scale":r["scale"],"fixture":r["fixture"],"status":r["status"]} for r in rows if r["limit"]==16 and r["mode"]==0 and not r["accepted"]],
 "decision":"Original and unused-rotation impacts and clear controls accept at cap16. Thinner impact stresses remain rejected/restored; all-impact resolution is incomplete.",
 "limits":["Hybrid CCD mode with the production iteration cap of 16; production default CCD mode is speculative.",
 "Zero impulse is a failure here because a force-free principal-axis box sweep analytically crosses the fixed plane.",
 "Positive endpoint gap cannot certify the transient sweep.","Wall times include initialization and are not performance comparisons."]},indent=2))
