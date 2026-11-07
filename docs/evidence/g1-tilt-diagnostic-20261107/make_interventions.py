"""Predeclared open-loop perturbations; requires a matching fresh policy replay."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("baseline_actions", type=Path)
p.add_argument("baseline_trace", type=Path)
p.add_argument("baseline_result", type=Path)
p.add_argument("output_prefix", type=Path)
args = p.parse_args()
raw = args.baseline_actions.read_bytes()
assert len(raw) == 1000 * 8 * 29 * 4
values = list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
assert all(math.isfinite(v) for v in values)
result = json.loads(args.baseline_result.read_text())
assert result["failed_environment_steps"] == 0
terminal = result["first_termination_step_by_environment"][0]
trace_lines = args.baseline_trace.read_text().splitlines()
assert trace_lines[0].endswith("environment=0") and "timestep=0.02" in trace_lines[0]
rows = [line.split() for line in trace_lines if not line.startswith("#")]
assert len(rows) == 1000 and int(rows[0][0]) == 1
angles = []
for i, row in enumerate(rows[:terminal]):
    assert int(row[0]) == i + 1
    qx, qy, qz, qw = map(float, row[4:8])
    gx, gy = 2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw)
    gz = 1-2*(qx*qx+qy*qy)
    angles.append(math.atan2(math.hypot(gx, gy), max(gz, 1e-6)))
assert abs(angles[-1]-result["first_termination_tilt_rad_by_environment"][0]) < 1e-5
# Trigger is derived only from the original fresh baseline. Actions start on
# the following control step. The baseline must survive the complete pulse.
trigger = next((i+1 for i in range(4, len(angles)) if all(t > 0.5 for t in angles[i-4:i+1])), None)
if trigger is None or trigger + 10 >= terminal:
    raise SystemExit("no eligible 10-step preterminal window; retain as no intervention")
manifest = {"trigger_step": trigger, "first_action_step": trigger+1,
            "last_action_step": trigger+10, "environment": 0,
            "joint": "left_ankle_roll_joint", "action_index": 5,
            "baseline_sha256": hashlib.sha256(raw).hexdigest(), "arms": {}}
prepared = []
for name, delta in (("positive", 0.1), ("negative", -0.1)):
    arm = values.copy()
    changed = []
    for step in range(trigger + 1, trigger + 11):
        index = ((step-1)*8)*29 + 5
        v = arm[index] + delta
        # Conservative admissibility guard. Do not silently clip a treatment.
        if not -1.0 <= v <= 1.0:
            raise SystemExit("action would leave [-1,1]; retain as no intervention")
        arm[index] = v
        changed.append(index)
    data = struct.pack("<" + "f" * len(arm), *arm)
    path = args.output_prefix.with_name(args.output_prefix.name + "-" + name + ".f32")
    prepared.append((path, data))
    manifest["arms"][name] = {"delta": delta, "indices": changed, "sha256": hashlib.sha256(data).hexdigest()}
for path, data in prepared:
    path.write_bytes(data)
print(json.dumps(manifest, indent=2, sort_keys=True))
