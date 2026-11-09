#!/usr/bin/env python3
"""Independent local representability of archived slider momentum equations."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import struct

E = Path(__file__).resolve().parents[2] / "docs/evidence/midpoint-first-failure-20261109"
NAMES = ("slider-primary-h15625-001", "slider-near-h15625-001",
         "slider-primary-h078125-001", "slider-near-h078125-001")

def fp32(x):
    return struct.unpack("!f", struct.pack("!f", float(x)))[0]

def exact(x):
    return Decimal.from_float(float(x))

def nearest_fp32(value):
    """Round a finite Decimal to binary32, checking adjacent candidates exactly."""
    center = struct.unpack("!I", struct.pack("!f", fp32(value)))[0]
    bits = {center, max(0, center - 1), min(0xffffffff, center + 1)}
    if center in (0, 0x80000000):
        bits.update((0x00000001, 0x80000001))
    candidates = []
    for word in bits:
        if (word & 0x7f800000) == 0x7f800000:
            continue
        number = struct.unpack("!f", struct.pack("!I", word))[0]
        candidates.append((abs(exact(number) - value), word & 1, number))
    return min(candidates)[2]

def analyze(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    mass, h = exact(data["slider_mass_kg"]), exact(data["h"])
    gravity = exact(fp32(-9.81))
    old = exact(data["initial_slider_velocity_mps"])
    result = {"trace": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
              "accepted_steps": 0, "single_exceeds_gate": 0, "split_exceeds_gate": 0,
              "single_max_residual_Ns": 0., "split_max_residual_Ns": 0.,
              "single_max_gate_ratio": 0., "split_max_gate_ratio": 0.}
    for step in data["steps"]:
        if not step["accepted"]:
            continue
        impulse = exact(step["owner_wrench_impulse"])
        ideal = old + h * gravity + impulse / mass
        high = nearest_fp32(ideal)
        low = nearest_fp32(ideal - exact(high))
        limit = max(Decimal("1e-12"), Decimal("1e-5") * abs(exact(step["fem_impulse"])))
        single = abs(mass * (exact(high) - old) - mass * h * gravity - impulse)
        split = abs(mass * (exact(high) + exact(low) - old) - mass * h * gravity - impulse)
        result["accepted_steps"] += 1
        for name, residual in (("single", single), ("split", split)):
            result[name + "_exceeds_gate"] += int(residual > limit)
            result[name + "_max_residual_Ns"] = max(result[name + "_max_residual_Ns"], float(residual))
            result[name + "_max_gate_ratio"] = max(result[name + "_max_gate_ratio"], float(residual / limit))
        old = exact(step["v"])
    return result

if __name__ == "__main__":
    with localcontext() as ctx:
        ctx.prec = 70
        print(json.dumps({"schema": "numi.slider-split-representability.v1",
                          "method": "Exact decimal evaluation of stored binary inputs; ideal endpoint split into nearest binary32 high plus nearest binary32 remainder independently at each recorded starting velocity.",
                          "limitations": "Historical measured forces are held fixed. This does not run or qualify a native transaction, Metal arithmetic, contact feedback, work, replay, or position.",
                          "traces": [analyze(E / (name + ".json")) for name in NAMES]}, indent=2, sort_keys=True))
