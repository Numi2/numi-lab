#!/usr/bin/env python3
"""Native full-step precontact regression at the previously rejected state."""
import json
import subprocess
import sys

probe, shader = sys.argv[1:]
run = subprocess.run([probe, shader, "0.00000390625", "204", "8",
    "ipc_area_midpoint", "0.0002", "physical", "contact", "reference"],
    text=True, capture_output=True, check=False, timeout=180)
if run.returncode:
    raise SystemExit("native step rejected: " + run.stderr[-3000:])
d = json.loads(run.stdout)
limit = d["mechanical_absolute_tolerance_Ns"] + (
    d["mechanical_relative_tolerance"] * d["mechanical_impulse_scale_Ns"])
assert d["accepted_steps"] == 204 and d["rejected_steps"] == 0
assert all(t["accepted"] and t["status"] == 0 for t in d["traces"])
assert d["traces"][-1]["independent_residual_l2_Ns"] <= limit
assert d["independent_max_local_scaled_without_floor"] <= .005
assert d["minimum_gap"] > 1e-7
assert d["traces"][-1]["force"] == 0
assert d["forced_rollback_exact"] and d["restore_replay"] and d["continuation_replay"]
print(json.dumps({"accepted_steps":d["accepted_steps"],
    "last_residual_Ns":d["traces"][-1]["independent_residual_l2_Ns"],
    "limit_Ns":limit,"physical_sha256":d["final_physical_sha256"],
    "rollback_and_replay":True}))
