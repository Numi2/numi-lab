#!/usr/bin/env python3
"""Exercise policy phase and rate limit across resident submission boundaries."""
import json
import math
import os
import subprocess
import sys

probe, shader = sys.argv[1:]
reference = None
for mode in ("normal", "ABORT_GROW", "ABORT_FRESH"):
    environment = os.environ.copy()
    environment.pop("ABORT_GROW", None)
    environment.pop("ABORT_FRESH", None)
    if mode != "normal":
        environment[mode] = "1"
    result = subprocess.run(
        [probe, shader], env=environment, text=True, capture_output=True
    )
    if result.returncode:
        raise AssertionError(f"{mode}: probe failed: {result.stderr}")
    runs = json.loads(result.stdout)["runs"]
    assert len(runs) == 6
    for run in runs:
        assert run["statuses"] == [0, 0, 0]
        assert run["physics_errors"] == [0, 0, 0]
        assert run["resets"] == [1, 0, 0]
        assert len(run["phases"]) == len(run["physical0"]) == 3
        assert all(math.isfinite(x) for x in run["q"] + run["v"])
        for actual, expected in zip(run["phases"], (0.02, 0.04, 0.06)):
            assert math.isclose(actual, expected, abs_tol=1e-7), (mode, run)
        for actual, expected in zip(run["physical0"], (0.005, 0.01, 0.015)):
            assert math.isclose(actual, expected, abs_tol=1e-7), (mode, run)
        value = {k: v for k, v in run.items() if k not in ("sizes", "repeat")}
        if reference is None:
            reference = value
        assert value == reference, (mode, run["sizes"], run["repeat"])
    if mode != "normal":
        assert "aborted init=" in result.stderr
print("stateful policy resident capacity and abort replay passed")
