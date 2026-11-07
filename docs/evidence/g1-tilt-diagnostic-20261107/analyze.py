"""Inspect the three retained first-episode traces; no inference of mechanical cause."""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "g1-episode-local-20261107"
results = []
for label, env in (("policy-4081007", 2), ("policy-4081008", 0), ("zero-4081007", 2)):
    raw = (root / (label + ".trace.tsv")).read_bytes()
    record = json.loads((root / (label + ".stdout")).read_text())
    terminal = record["first_termination_step_by_environment"][env]
    rows = []
    for line in raw.decode().splitlines():
        if line.startswith("#"):
            continue
        row = [float(x) for x in line.split()]
        step = int(row[0])
        assert step == len(rows) + 1
        if step > terminal:
            break
        x, y, z, qx, qy, qz, qw = row[1:8]
        gx = 2 * (qx * qz - qy * qw)
        gy = 2 * (qy * qz + qx * qw)
        gz = 1 - 2 * (qx*qx + qy*qy)
        angle = math.atan2(math.hypot(gx, gy), max(gz, 1e-6))
        rows.append(dict(step=step, x=x, y=y, z=z, tilt=angle))
    assert len(rows) == terminal
    reported = record["first_termination_tilt_rad_by_environment"][env]
    assert abs(rows[-1]["tilt"] - reported) < 1e-5
    def first_above(threshold):
        return next((row for row in rows if row["tilt"] > threshold), None)
    def final_sustained_above(threshold):
        if rows[-1]["tilt"] <= threshold:
            return None
        idx = len(rows) - 1
        while idx and rows[idx - 1]["tilt"] > threshold:
            idx -= 1
        return rows[idx]
    results.append(dict(arm=label, environment=env, trace_sha256=hashlib.sha256(raw).hexdigest(), first_post_step=rows[0], first_above_0_5_rad=first_above(0.5), final_contiguous_above_0_5_rad=final_sustained_above(0.5), first_above_1_0_rad=first_above(1.0), terminal=rows[-1], terminal_tilt_reported=reported))
print(json.dumps(results, indent=2, sort_keys=True))
