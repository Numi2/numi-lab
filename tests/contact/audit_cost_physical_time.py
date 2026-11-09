#!/usr/bin/env python3
"""Recompute timed rollout rates using the actual binary32 device timestep."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from cost_summary import summarize


def binary32(value):
    return struct.unpack("=f", struct.pack("=f", value))[0]


def audit(directory, tag):
    directory = Path(directory)
    if not tag.replace("-", "").isalnum():
        raise ValueError("invalid study tag")
    expected = [("ipc_area", 1), ("ipc_area_midpoint", 1),
                ("ipc_area_midpoint", 2), ("ipc_area", 2),
                ("ipc_area", 3), ("ipc_area_midpoint", 3)]
    corrected, sources = [], []
    for arm, repetition in expected:
        stem = directory / (tag + "-" + arm + "-r" + str(repetition))
        result_path, record_path = Path(str(stem) + ".json"), Path(str(stem) + "-execution.json")
        result, record = json.loads(result_path.read_text()), json.loads(record_path.read_text())
        requested, actual = result["h"], binary32(result["h"])
        steps, wall = result["accepted_steps"], result["wall_seconds"]
        if (record.get("arm") != arm or record.get("repetition") != repetition or
            not record.get("accepted_and_matches_accuracy_state") or result.get("arm") != arm or
            result.get("rejected_steps") != 0 or steps != result.get("requested_steps") or
            not math.isfinite(requested) or requested <= 0 or not math.isfinite(actual) or actual <= 0 or
            not math.isfinite(wall) or wall <= 0 or
            not math.isclose(result["accepted_simulated_seconds"], steps * requested, rel_tol=1e-12) or
            not math.isclose(result["accepted_seconds_per_wall_second"], steps * requested / wall, rel_tol=1e-6)):
            raise ValueError("invalid cost rollout " + str(stem))
        corrected.append({"arm": arm, "repetition": repetition, "accepted_and_matches_accuracy_state": True,
                          "accepted_seconds_per_wall_second": steps * actual / wall})
        sources.append({"arm": arm, "repetition": repetition, "requested_h": requested,
                        "device_h_binary32": actual, "device_accepted_seconds": steps * actual,
                        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                        "execution_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest()})
    return {"device_time_summary": summarize(corrected), "rollouts": sources,
            "meaning": "binary32 device time correction only; accuracy and strict trace gates remain independent"}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_cost_physical_time.py EVIDENCE_DIR TAG")
    try:
        print(json.dumps(audit(sys.argv[1], sys.argv[2]), indent=2, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        print(json.dumps({"valid": False, "error": repr(error)}))
        raise SystemExit(1)
