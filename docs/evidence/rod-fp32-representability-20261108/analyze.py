#!/usr/bin/env python3
"""Exact finite-lattice test for the retained straight-rod scalar equations."""
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys

here = Path(__file__).resolve().parent
source = here.parent / "rod-local-momentum-20261108/analysis-v3.json"
plan = here / "PLAN.md"
records = json.loads(source.read_text())["cases"]
matches = [x for x in records if x["arm"] == "guarded" and x["n"] == 8
           and x["axis"] == "x" and x["offset"] == 0.25 and x["kind"] == "fixed"]
assert len(matches) == 1
d = matches[0]
assert d["accepted"] and d["axis_v"] == [1, 0, 0] and d["pinned"] == [1,0,0,0,0,0,0,1]
n, h = d["n"], d["h"]
old = [d["origin"][0] + x for x in d["x0"]]
v0, masses, lengths, ea = d["v0"], d["mass"], d["lengths"], d["ea"]
k = [a / length for a, length in zip(ea, lengths)]
assert all(math.isfinite(x) for x in old + v0 + masses + lengths + ea)
assert min(masses) > 0 and min(lengths) > 0 and max(k) - min(k) < 1e-10
assert all(all(abs(p[j] - d["origin"][j]) < 1e-14 for j in (1, 2))
           for p in d["positions"])
pins = [d["positions"][0][0], d["positions"][-1][0]]
reference_pins = [old[0], old[-1]]

# Solve A*x=b via Gaussian elimination, pivoted in FP64. The reference is
# assembled directly from the physical spring energy and original input.
m = n - 2
matrix, rhs = [], []
for i in range(1, n - 1):
    row = [0.0] * m
    row[i - 1] = masses[i] / h**2 + k[i-1] + k[i]
    b = masses[i] / h**2 * (old[i] + h*v0[i]) + k[i-1]*lengths[i-1] - k[i]*lengths[i]
    if i > 1: row[i-2] = -k[i-1]
    else: b += k[i-1] * reference_pins[0]
    if i < n-2: row[i] = -k[i]
    else: b += k[i] * reference_pins[1]
    matrix.append(row); rhs.append(b)
for j in range(m):
    pivot = max(range(j, m), key=lambda t: abs(matrix[t][j]))
    assert abs(matrix[pivot][j]) > 1e-20
    matrix[j], matrix[pivot] = matrix[pivot], matrix[j]
    rhs[j], rhs[pivot] = rhs[pivot], rhs[j]
    for i in range(j+1, m):
        factor = matrix[i][j]/matrix[j][j]
        for t in range(j, m): matrix[i][t] -= factor*matrix[j][t]
        rhs[i] -= factor*rhs[j]
xint = [0.0]*m
for i in range(m-1, -1, -1):
    xint[i] = (rhs[i]-sum(matrix[i][j]*xint[j] for j in range(i+1,m)))/matrix[i][i]
xref = [reference_pins[0], *xint, reference_pins[1]]
vref = [(x-o)/h for x,o in zip(xref,old)]

def gradient(i, left, center, right):
    assert left < center < right, "edge reversal invalidates linearized straight model"
    return k[i-1]*(center-left-lengths[i-1]) - k[i]*(right-center-lengths[i])

def velocity_points(i, left, center, right):
    return ((center-old[i])/h,
            v0[i] - h/masses[i]*gradient(i,left,center,right),
            vref[i])

def cost(i, left, center, right):
    values = velocity_points(i, left, center, right)
    return (max(values)-min(values))/2

ref_residual = max(abs(vref[i]-v0[i] +
                   h/masses[i]*gradient(i,*xref[i-1:i+2])) for i in range(1,n-1))
assert ref_residual < 1e-8
captured = [p[0] for p in d["positions"]]
oracle_error = max(abs(d["velocities"][i][0]-v0[i] +
                       h/masses[i]*gradient(i,*captured[i-1:i+2]) -
                       d["localres5"][i][0]) for i in range(1,n-1))
assert oracle_error < 1e-4

def f32_bits(value):
    return struct.unpack(">I", struct.pack(">f", value))[0]
def from_bits(value):
    return struct.unpack(">f", struct.pack(">I", value))[0]

position_gate = 1e-6
def world_candidates(value):
    center = f32_bits(value)
    candidates = []
    # All values in this fixture are positive normal floats with constant ulp.
    assert value-position_gate > 0.25 and value+position_gate < 0.5
    for bits in range(center-100, center+101):
        x = from_bits(bits)
        if abs(x-value) <= position_gate: candidates.append(x)
    assert candidates and abs(from_bits(center-101)-value) > position_gate
    assert abs(from_bits(center+101)-value) > position_gate
    return candidates

def local_candidates(value):
    # Constructive finite subset, not a global optimum.
    origin = d["origin"][0]
    center = f32_bits(value-origin)
    return [origin+from_bits(b) for b in range(center-4,center+5)
            if abs(origin+from_bits(b)-value) <= position_gate]

def solve_dp(candidates, eval_cost):
    states = {(0,j): (0.0, (0,j)) for j in range(len(candidates[1]))}
    counts = []
    for i in range(1,len(candidates)-1):
        new = {}
        for (a,b),(prev,path) in states.items():
            left, center = candidates[i-1][a], candidates[i][b]
            for c,right in enumerate(candidates[i+1]):
                v = max(prev, eval_cost(i,left,center,right))
                key=(b,c)
                if key not in new or v < new[key][0]:
                    new[key]=(v,path+(c,))
        assert new
        states=new
        counts.append(len(states))
    return min(states.values(), key=lambda z:z[0]), counts

# Calibrate the minimax DP independently using a deliberately asymmetric
# discrete chain where exhaustive Cartesian enumeration is small.
cal = [[0.0],[-2.0,0.3,1.9],[-1.2,0.8,2.2],[2.5]]
cal_cost = lambda i,a,b,c: abs(.1*a-.3*b+.2*c+.01*i)
(dp, _), _ = solve_dp(cal, cal_cost)
brute = min(max(cal_cost(i,*path[i-1:i+2]) for i in range(1,len(path)-1))
            for path in itertools.product(*cal))
assert abs(dp-brute) < 1e-15

world = [[pins[0]]] + [world_candidates(x) for x in xref[1:-1]] + [[pins[1]]]
local = [[pins[0]]] + [local_candidates(x) for x in xref[1:-1]] + [[pins[1]]]
outputs = {}
for name, candidates in (("world_exhaustive",world),("local_subset",local)):
    (opt,path), states = solve_dp(candidates,cost)
    positions = [xs[index] for xs,index in zip(candidates,path)]
    assert all(positions[i] < positions[i+1] for i in range(n-1))
    triplets = [velocity_points(i,*positions[i-1:i+2]) for i in range(1,n-1)]
    velocities = [(min(t)+max(t))/2 for t in triplets]
    outputs[name] = {"minimum_max_error_m_per_s":opt,
        "passes_0_001_m_per_s_gate":opt <= .001,
        "candidate_counts":[len(c) for c in candidates], "dp_state_counts":states,
        "witness_positions":positions,"witness_free_velocities":velocities,
        "max_position_error_m":max(abs(a-b) for a,b in zip(positions,xref)),
        "exhaustive":name=="world_exhaustive"}
result={"plan_sha256":hashlib.sha256(plan.read_bytes()).hexdigest(),
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python":sys.version,"reference_equation_residual_m_per_s":ref_residual,
        "analytic_vs_energy_oracle_m_per_s":oracle_error,
        "dp_vs_bruteforce_absolute_difference":abs(dp-brute),
        "reference_positions":xref,"reference_velocities":vref,"arms":outputs}
(here/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,indent=2,sort_keys=True))
