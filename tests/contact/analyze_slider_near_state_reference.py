#!/usr/bin/env python3
"""Time-refinement check of near-contact slider endpoint state."""
import hashlib
import json
from pathlib import Path
import sys

e = Path(__file__).resolve().parents[2] / "docs/evidence/midpoint-first-failure-20261109"
receipt = json.loads((e / "slider-work-history-reference-002-execution.json").read_text())
states = []
hashes = {}
for label, i in (("050", 2), ("025", 3)):
    name = "slider-near-rk4-h" + label + ".json"
    raw = (e / name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != receipt["runs"][i]["historical_sha256"] or not receipt["runs"][i]["historic_fields_exact_match"]:
        raise ValueError("reference state does not match retained reference receipt")
    x = json.loads(raw)
    if x["initial_gap_m"] != .000095:
        raise ValueError("wrong near-contact initial gap")
    states.append(x)
    hashes[name] = digest
a, b = states
rows = {}
for key, name, floor in (("slider_final_z_m", "position", 1e-8),
                         ("slider_final_v_mps", "velocity", 1e-6)):
    denominator = max(abs(b[key]), floor)
    error = abs(a[key] - b[key]) / denominator
    rows[name] = {"absolute_error": abs(a[key] - b[key]), "normalization": denominator,
                  "relative_error": error, "passes": error <= .0025}
out = {"schema": "numi.slider-near-state-reference.v1", "inputs_sha256": hashes,
       "state": rows, "passes": all(row["passes"] for row in rows.values()),
       "scope": "Independent FP64 endpoint temporal refinement only; no native rollout qualification."}
print(json.dumps(out, indent=2, sort_keys=True))
raise SystemExit(0 if out["passes"] else 1)
