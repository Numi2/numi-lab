#!/usr/bin/env python3
"""Compare a recorded native trajectory against the predeclared strand samples."""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import screen_ophthalmic_strand as screen

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workbook", type=Path)
    p.add_argument("trajectory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    series = screen.parse(a.workbook)
    raw = a.trajectory.read_bytes()
    with a.trajectory.open(newline="") as f:
        rows = list(csv.DictReader(f))
    xs, forces = [], []
    for r in rows:
        if r["status"] != "ok":
            raise ValueError("trajectory contains rejected sample")
        x, force = float(r["extension_mm"]), float(r["force_N"])
        if not math.isfinite(x) or not math.isfinite(force) or (xs and x <= xs[-1]):
            raise ValueError("invalid native trajectory")
        xs.append(x)
        forces.append(force)
    if len(xs) < 2 or xs[0] > 2.0 or xs[-1] < 4.0:
        raise ValueError("native trajectory lacks full interval")
    results = []
    for name, data in zip(screen.NAMES[1:], series[1:]):
        if len(data) < 20:
            raise ValueError("insufficient held-out samples")
        numerator = denominator = 0.0
        for x, observed in data:
            i = bisect.bisect_left(xs, x)
            if i == 0:
                predicted = forces[0]
            elif i == len(xs):
                raise ValueError("sample outside trajectory")
            else:
                predicted = forces[i-1] + (forces[i]-forces[i-1])*(x-xs[i-1])/(xs[i]-xs[i-1])
            numerator += abs(predicted-observed)
            denominator += abs(observed)
        if denominator == 0:
            raise ValueError("zero held-out force denominator")
        error = numerator / denominator
        results.append({"specimen": name, "count": len(data),
                        "relative_absolute_error": error, "passed": error <= .1})
    result = {"workbook_sha256": screen.EXPECTED_SHA,
              "trajectory_sha256": hashlib.sha256(raw).hexdigest(),
              "heldout": results, "passed": all(r["passed"] for r in results)}
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
