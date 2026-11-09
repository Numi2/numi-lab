#!/usr/bin/env python3
"""Time-refinement check of slider contact torque impulses."""
import hashlib
import json
from pathlib import Path
import sys

e = Path(__file__).resolve().parents[2] / "docs/evidence/midpoint-first-failure-20261109"
receipt = json.loads((e / "slider-work-history-reference-002-execution.json").read_text())
result = {"schema": "numi.slider-torque-reference.v1", "fixtures": {},
          "tolerance": .01, "absolute_scale_Nms": 1e-12,
          "scope": "Independent FP64 torque temporal refinement only; not native qualification."}
for fixture, prefix, offset in (("primary", "slider-v2-rk4-h", 0),
                                ("near", "slider-near-rk4-h", 2)):
    states, hashes = [], {}
    for label, index in (("050", offset), ("025", offset + 1)):
        name = prefix + label + ".json"
        raw = (e / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        run = receipt["runs"][index]
        if digest != run["historical_sha256"] or not run["historic_fields_exact_match"]:
            raise ValueError("unbound historical torque reference")
        states.append(json.loads(raw)); hashes[name] = digest
    coarse, fine = states
    axes = {}
    for axis in ("x", "y"):
        key = "torque_impulse_" + axis + "_Nms"
        error = abs(coarse[key] - fine[key])
        scale = max(abs(fine[key]), 1e-12)
        axes[axis] = {"absolute_error_Nms": error, "reference_scale_Nms": scale,
                      "normalized_error": error / scale, "passes": error / scale <= .01}
    result["fixtures"][fixture] = {"inputs_sha256": hashes, "axes": axes,
                                  "passes": all(x["passes"] for x in axes.values())}
result["passes"] = all(x["passes"] for x in result["fixtures"].values())
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["passes"] else 1)
