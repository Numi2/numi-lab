#!/usr/bin/env python3
"""Independently validate reference trace geometry and numerical invariants.

This is an input-validity check, not a temporal or spatial convergence test.
"""
import argparse, hashlib, json, math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("trace", type=Path)
parser.add_argument("divisions", type=int)
parser.add_argument("h", type=float)
a = parser.parse_args()
with a.trace.open("rb") as stream:
    sha = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib,"file_digest") else None
if sha is None:
    sha = hashlib.sha256(a.trace.read_bytes()).hexdigest()
x = json.loads(a.trace.read_text())
n = a.divisions
if not (x["mesh_divisions"] == n and x["h"] == a.h and x["duration"] == .01 and
        x["nodes"] == (n+1)**3 and x["tetrahedra"] == 6*n**3):
    raise ValueError("reference grid or time interval mismatch")
weights = x["contact_area_weights_m2"]
if len(weights) != (n+1)**3:
    raise ValueError("contact area weight count mismatch")
expected = [0.0] * (n+1)**2
share = .02*.02/(n*n*6.)
for y in range(n):
    for i in range(n):
        p = y*(n+1)+i
        for j in (p,p+1,p+n+2,p,p+n+2,p+n+1):
            expected[j] += share
area_tol = .0004*1e-12
for i, value in enumerate(weights):
    target = expected[i] if i < len(expected) else 0.0
    if not math.isfinite(value) or value < 0 or abs(value-target) > area_tol:
        raise ValueError("invalid reference area weight at " + str(i))
area_sum = math.fsum(weights)
if abs(area_sum-.0004) >= 1e-14 or abs(x["area_m2"]-.0004) >= 1e-14:
    raise ValueError("incorrect total contact area")
if abs(x["mass_kg"]-.0088) >= 1e-8 or not math.isfinite(x["min_gap_m"]) or x["min_gap_m"] <= 1e-7:
    raise ValueError("invalid mass or clearance")
trace = x["trace"]
if len(trace) != 5001 or len(x["force_bins_N"]) != 10:
    raise ValueError("invalid trace length")
previous = -1.0
for i,row in enumerate(trace):
    if len(row) != 3 or not all(math.isfinite(v) for v in row):
        raise ValueError("invalid force trace")
    if abs(row[0]-i*.000002) > 1e-12 or row[1] < 0 or row[2] < previous-1e-15:
        raise ValueError("invalid force trace time, sign, or accumulated impulse")
    previous = row[2]
if not all(math.isfinite(v) and v >= 0 for v in x["force_bins_N"]):
    raise ValueError("invalid force bins")
if not (math.isfinite(x["peak_force_N"]) and x["peak_force_N"] > 0 and
        math.isfinite(x["impulse_Ns"]) and x["impulse_Ns"] > 0 and
        abs(x["impulse_Ns"]-trace[-1][2]) < 1e-12):
    raise ValueError("invalid peak or impulse")
print(json.dumps({"scope":"reference input validation only; no temporal or spatial convergence claim",
                  "path":str(a.trace),"sha256":sha,"mesh_divisions":n,"h":a.h,
                  "area_m2":area_sum,"weights_checked":len(weights),"trace_samples_checked":len(trace),
                  "peak_force_N":x["peak_force_N"],"impulse_Ns":x["impulse_Ns"],"passes":True},indent=2))
