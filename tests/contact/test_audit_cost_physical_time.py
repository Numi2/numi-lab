#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
from audit_cost_physical_time import audit, binary32

assert binary32(.000001) != .000001
schedule = [("ipc_area", 1), ("ipc_area_midpoint", 1), ("ipc_area_midpoint", 2),
            ("ipc_area", 2), ("ipc_area", 3), ("ipc_area_midpoint", 3)]
with tempfile.TemporaryDirectory() as d:
    p = Path(d)
    for arm, rep in schedule:
        h = .000001 if arm == "ipc_area" else .000015625
        steps = round(.01 / h)
        stem = p / ("study-" + arm + "-r" + str(rep))
        result = {"arm": arm, "h": h, "accepted_steps": steps, "requested_steps": steps,
                  "rejected_steps": 0, "wall_seconds": 1., "accepted_simulated_seconds": steps*h,
                  "accepted_seconds_per_wall_second": steps*h}
        record = {"arm": arm, "repetition": rep, "accepted_and_matches_accuracy_state": True}
        Path(str(stem)+".json").write_text(json.dumps(result))
        Path(str(stem)+"-execution.json").write_text(json.dumps(record))
    output = audit(p, "study")
    assert output["device_time_summary"]["complete"]
    assert output["rollouts"][0]["device_accepted_seconds"] != .01
    target = p / "study-ipc_area-r1.json"
    broken = json.loads(target.read_text()); broken["rejected_steps"] = 1;target.write_text(json.dumps(broken))
    try:
        audit(p, "study")
    except ValueError:
        pass
    else:
        raise AssertionError("rejected rollout was accepted")
print("binary32 physical-time and rejection checks passed")
