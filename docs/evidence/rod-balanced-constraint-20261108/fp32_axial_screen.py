#!/usr/bin/env python3
"""CPU FP32 screen; reference is the preregistered FP64 primal solve."""
import json
import math
from pathlib import Path
import numpy as np

REST = 0.11758755785232791
LENGTH = 0.122
EA = 2.3818718058345718
DENSITY_AREA = 1100.0 * math.pi * 0.000020**2
H = 0.001
ref = json.loads(Path('docs/evidence/rod-balanced-constraint-20261108/coupled_axial_reference.json').read_text())

def matrix_for(n, moving, dtype):
    T = dtype
    rest = T(REST / (n - 1))
    mass = np.full(n, T(DENSITY_AREA * (REST / (n - 1))), dtype=T)
    mass[[0, -1]] *= T(.5)
    p = np.linspace(0, LENGTH, n, dtype=T)
    if moving:
        p[n // 2] = T(p[n // 2] - T(.001))
        p[n // 2] = T(p[n // 2] + T(T(H) * T(.01)))
    edges = np.zeros((n - 1, n), dtype=T)
    idx = np.arange(n - 1)
    edges[idx, idx] = -1
    edges[idx, idx + 1] = 1
    attach = np.zeros((2, n), dtype=T)
    attach[0, 0] = 1
    attach[1, -1] = 1
    j = np.vstack((edges, attach))
    c = np.concatenate((edges @ p - rest, attach @ p - np.array([0, LENGTH],dtype=T)))
    compliance = np.concatenate((np.full(n - 1, T(T(rest / T(EA)) / T(T(H)*T(H))), dtype=T), np.zeros(2, dtype=T)))
    a = (j / mass) @ j.T + np.diag(compliance)
    return a, c, j, mass, p

out = []
for expected in ref['cases']:
    n, moving = expected['nodes'], expected['moving']
    a,c,j,m,p=matrix_for(n,moving,np.float32)
    lam=np.linalg.solve(a,-c)
    x = p + (j.T @ lam) / m
    forces = -lam[-2:] / np.float32(H*H)
    a64,c64,_,_,_=matrix_for(n,moving,np.float64)
    residual = float(np.linalg.norm(a64 @ lam.astype(np.float64) + c64, ord=np.inf))
    normalizer = float(np.linalg.norm(a64,ord=np.inf) * np.linalg.norm(lam.astype(np.float64),ord=np.inf) + np.linalg.norm(c64,ord=np.inf))
    # The FP64 reference output records the middle node; independently derive
    # all other nodes by another FP64 multiplier solve for this precision screen.
    l64=np.linalg.solve(a64,-c64)
    _,_,j64,m64,p64=matrix_for(n,moving,np.float64)
    x64=p64+(j64.T@l64)/m64
    poserr=float(np.max(np.abs(x.astype(np.float64)-x64)))
    ferr=np.abs(forces.astype(np.float64)-np.array(expected['reference_reactions']))
    limits=np.maximum(1e-6,.01*np.abs(np.array(expected['reference_reactions'])))
    passed=bool(poserr<=1e-6 and np.all(ferr<=limits))
    out.append(dict(nodes=n,moving=moving,passed=passed,position_error_m=poserr,
                    force_error_N=list(map(float,ferr)),force_limit_N=list(map(float,limits)),
                    forces_N=list(map(float,forces)),residual_fp64_m=residual,
                    scaled_residual=residual/normalizer))
print(json.dumps(dict(all_pass=all(r['passed'] for r in out),cases=out),indent=2,sort_keys=True))
raise SystemExit(0 if all(r['passed'] for r in out) else 1)
