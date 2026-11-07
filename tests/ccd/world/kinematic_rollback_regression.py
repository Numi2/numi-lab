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
print(p.stdout)
