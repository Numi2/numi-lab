#!/usr/bin/env python3
"""Compare fixture-specific local impulse scales against retained trials."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "docs/evidence"
old_root = root / "midpoint-spatial-cost-dynamic-20261108"
new_root = root / "midpoint-first-failure-20261109"
out = {}
for n in (2, 4):
    old = json.loads((old_root / f"native-n{n}-ipc_area_midpoint-h015625.json").read_text())
    new = json.loads((new_root / f"scaled-n{n}.json").read_text())
    keys = ("gap", "height", "force", "contact_work_J", "accepted_normal_impulse", "momentum_z", "contacts", "independent_max_local_scaled_residual")
    mismatch = {k: sum(a[k] != b[k] for a, b in zip(old["traces"], new["traces"])) for k in keys}
    check = {
        "accepted_steps": new["accepted_steps"],
        "rejected_steps": new["rejected_steps"],
        "legacy_floor_max": new["independent_max_local_scaled_residual"],
        "physical_scale_max": new["independent_max_local_scaled_without_floor"],
        "physical_scale_limit": .005,
        "worst_step": max(new["traces"], key=lambda x: x["independent_max_local_scaled_without_floor"])["step"],
        "momentum_max_absolute_error_Ns": new["momentum_max_absolute_error_Ns"],
        "same_physical_digest": old["final_physical_sha256"] == new["final_physical_sha256"],
        "same_physics_fingerprint": old["source_fingerprint"] == new["source_fingerprint"],
        "same_device_fingerprint": old["device_fingerprint"] == new["device_fingerprint"],
        "trace_mismatches": mismatch,
        "restore_replay": new["restore_replay"],
        "continuation_replay": new["continuation_replay"],
    }
    check["passes_followup"] = (check["accepted_steps"] == 640 and check["rejected_steps"] == 0
        and len(old["traces"]) == len(new["traces"]) == 640
        and check["physical_scale_max"] <= check["physical_scale_limit"]
        and check["same_physical_digest"] and check["same_physics_fingerprint"]
        and check["same_device_fingerprint"] and not any(mismatch.values())
        and check["restore_replay"] and check["continuation_replay"])
    out[f"n{n}"] = check
print(json.dumps(out, indent=2, sort_keys=True))
sys.exit(0 if all(x["passes_followup"] for x in out.values()) else 1)
