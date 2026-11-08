#!/usr/bin/env python3
"""Compare actual mixed physical failure with and without policy restoration."""
import json
import math
import os
import subprocess
import sys

probe, shader = sys.argv[1:]
results = {}
for restore in (True, False):
    env = os.environ.copy()
    env.pop("NUMI_SKIP_POLICY_RESTORE", None)
    if not restore:
        env["NUMI_SKIP_POLICY_RESTORE"] = "1"
    runs = []
    for _ in range(2):
        command = subprocess.run([probe, shader], env=env, text=True, capture_output=True)
        if command.returncode:
            raise AssertionError(command.stderr)
        runs.append(json.loads(command.stdout))
    assert runs[0] == runs[1]
    results[restore] = runs[0]
    assert runs[0]["success"] == 27 and runs[0]["failed"] == 1
    failed, accepted = runs[0]["rows"]
    assert failed["world"] == failed["contact"] == 5 and failed["reset"] == 0
    assert accepted["world"] == accepted["contact"] == accepted["reset"] == 0
    assert math.isclose(accepted["next"] - accepted["original"], 0.005, abs_tol=1e-7)
    expected = 0.0 if restore else 0.005
    assert math.isclose(failed["next"] - failed["original"], expected, abs_tol=1e-7)
for field in ("q", "v"):
    assert results[True][field] == results[False][field]
assert results[True]["rows"][1] == results[False]["rows"][1]
print("actual mixed physical rejection preserves policy state for the failed environment")
