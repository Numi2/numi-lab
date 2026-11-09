#!/usr/bin/env python3
"""Check archived slider position representability against midpoint kinematics."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from analyze_slider_split_representability import E, NAMES, exact, nearest_fp32

def analyze(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    q0 = Decimal(0)
    v0 = exact(data["initial_slider_velocity_mps"])
    h = exact(data["h"])
    epsilon = Decimal(2) ** -23
    result = {"trace": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "accepted_steps": 0}
    for label in ("published", "single", "split"):
        result[label + "_exceeds_bound"] = 0
        result[label + "_maximum_defect_m"] = 0.
        result[label + "_maximum_bound_ratio"] = 0.
    for step in data["steps"]:
        if not step["accepted"]:
            continue
        v1 = exact(step["v"])
        dx = h * (v0 + v1) / 2
        ideal = q0 + dx
        high = nearest_fp32(ideal)
        low = nearest_fp32(ideal - exact(high))
        limit = 64 * epsilon * max(abs(dx), Decimal("1e-6"))
        for label, q1 in (("published", exact(step["q"])), ("single", exact(high)),
                          ("split", exact(high) + exact(low))):
            defect = abs(q1 - ideal)
            result[label + "_exceeds_bound"] += int(defect > limit)
            result[label + "_maximum_defect_m"] = max(result[label + "_maximum_defect_m"], float(defect))
            result[label + "_maximum_bound_ratio"] = max(result[label + "_maximum_bound_ratio"], float(defect / limit))
        result["accepted_steps"] += 1
        q0, v0 = exact(step["q"]), v1
    return result

if __name__ == "__main__":
    with localcontext() as context:
        context.prec = 70
        print(json.dumps({"schema": "numi.slider-position-representability.v1",
            "method": "Use stored endpoint velocities to reconstruct midpoint position; round the ideal local endpoint to one word and to a two-word expansion.",
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rounding_helper_sha256": hashlib.sha256((Path(__file__).parent / "analyze_slider_split_representability.py").read_bytes()).hexdigest(),
            "limitations": "Historical inputs are held fixed. This independently reconstructed bound is not the native stage-increment certificate; it does not qualify GPU arithmetic, contact feedback, accepted steps or rollback.",
            "traces": [analyze(E / (name + ".json")) for name in NAMES]}, indent=2, sort_keys=True))
