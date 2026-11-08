#!/usr/bin/env python3
"""Check environment growth and host-abort preservation of stateful policy."""
import json
import math
import os
import subprocess
import sys

all_runs = []
for without_reset in (False, True):
    environment = os.environ.copy()
    if without_reset:
        environment["NO_NEW_RESET"] = "1"
    else:
        environment.pop("NO_NEW_RESET", None)
    result = subprocess.run(sys.argv[1:], text=True, capture_output=True, check=True, env=environment)
    runs = json.loads(result.stdout)["runs"]
    assert len(runs) == 8
    all_runs.append(runs)
for without_reset, runs in enumerate(all_runs):
    for run in runs:
        continued = without_reset and run["initial"] == 2
        for key, expected in (("first_phase", .02), ("first_action", .005),
                              ("phase0", .04), ("phase1", .04 if continued else .02),
                              ("action0", .01), ("action1", .01 if continued else .005)):
            assert math.isclose(run[key], expected, rel_tol=0, abs_tol=1e-7), (key, run)
    for initial in (1, 2):
        for abort in (0, 1):
            pair = [r for r in runs if r["initial"] == initial and r["abort"] == abort]
            assert len(pair) == 2
            assert {k:v for k,v in pair[0].items() if k != "repeat"} == {k:v for k,v in pair[1].items() if k != "repeat"}
print("environment growth, reset, and pre-commit abort continuity passed")
