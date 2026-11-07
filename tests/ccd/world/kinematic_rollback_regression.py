#!/usr/bin/env python3
"""A rejected target restores populated resident state and clean continuation."""
import json, subprocess, sys
p=subprocess.run([sys.argv[1],sys.argv[2]],capture_output=True,text=True,check=True)
r=json.loads(p.stdout)
assert r["initial_ok"] and r["initial_cache_count"]==13 and r["initial_reference_identical"]
assert r["accepted_target_status"]==0 and r["accepted_target_applied"]
assert r["failure_status"]==4 and not r["failure_succeeded"]
assert r["failed_cache_count"]==13 and r["failed_state_restored"]
for key in ("restore_q","restore_v","restore_scene","restore_counts","restore_headers","restore_points",
            "continue_q","continue_v","continue_scene","continue_counts","continue_headers","continue_points"):
 assert r[key],key
assert r["continuation_status"]==r["control_status"]==0 and r["continuation_identical"]
# Reset is the pre-step boundary: compare live state and recontact, while
# retaining (not masking) the diagnostic's inactive-cache byte mismatch.
assert r["reset_initial_ok"] and r["reset_failure_status"]==4 and r["reset_control_status"]==0
assert r["reset_scene_x"]==2 and r["reset_cache_count"]==0
for key in ("reset_q_equal","reset_v_equal","reset_scene_equal","reset_counts_equal"):
 assert r[key],key
assert r["reset_continuation_status"]==r["reset_reference_status"]==0
assert r["reset_recontact_cache_count"]==r["reset_reference_recontact_cache_count"]==13
assert r["reset_continuation_identical"]
print(p.stdout)
