#!/usr/bin/env python3
"""Compare coupled axial multiplier and free-node stiffness equations in FP64."""
import json
import math
import numpy as np

H = 0.001
TOTAL_REST = 0.11758755785232791
LENGTH = 0.122
EA = 2.3818718058345718
AREA_DENSITY = 1100.0 * math.pi * 0.000020**2

def solve(n, moving):
    rest = TOTAL_REST / (n - 1)
    edge_mass = AREA_DENSITY * rest
    mass = np.full(n, edge_mass)
    mass[[0, -1]] *= .5
    start = np.linspace(0, LENGTH, n)
    velocity = np.zeros(n)
    if moving:
        start[n // 2] -= .001
        velocity[n // 2] = .01
    predicted = start + H * velocity
    edge_j = np.zeros((n - 1, n))
    idx = np.arange(n - 1)
    edge_j[idx, idx] = -1
    edge_j[idx, idx + 1] = 1
    attach_j = np.zeros((2, n))
    attach_j[0, 0] = 1
    attach_j[1, -1] = 1
    j = np.vstack((edge_j, attach_j))
    constraint = np.concatenate((edge_j @ predicted - rest,
                                 attach_j @ predicted - [0, LENGTH]))
    compliance = np.concatenate((np.full(n - 1, rest / EA / H**2), np.zeros(2)))
    matrix = (j / mass) @ j.T + np.diag(compliance)
    lam = np.linalg.solve(matrix, -constraint)
    position = predicted + (j.T @ lam) / mass
    forces = -lam[-2:] / H**2
    # Independently solve the primal free-node implicit stiffness equations.
    stiffness = EA / rest
    k = stiffness * edge_j.T @ edge_j
    operator = k + np.diag(mass / H**2)
    rhs = (mass / H**2) * predicted + stiffness * edge_j.T @ np.full(n - 1, rest)
    target = np.array([0., LENGTH])
    reference = np.empty(n)
    reference[[0, -1]] = target
    reference[1:-1] = np.linalg.solve(operator[1:-1, 1:-1],
                                      rhs[1:-1] - operator[1:-1][:, [0, -1]] @ target)
    ref_forces = np.array([stiffness * (reference[1] - reference[0] - rest),
                           -stiffness * (reference[-1] - reference[-2] - rest)])
    position_error = float(np.max(np.abs(position - reference)))
    force_error = float(np.max(np.abs(forces - ref_forces)))
    residual = float(np.max(np.abs(matrix @ lam + constraint)))
    passed = position_error <= 1e-10 and force_error <= 1e-8 and residual <= 1e-9
    return dict(nodes=n, moving=moving, accepted=bool(passed),
                position_error_m=position_error, reaction_error_N=force_error,
                multiplier_residual_m=residual, condition_number=float(np.linalg.cond(matrix)),
                middle_position=float(position[n // 2]),
                reactions=list(map(float, forces)),
                reference_reactions=list(map(float, ref_forces)))

cases = [solve(n, moving) for n in (3, 5, 9, 17) for moving in (False, True)]
print(json.dumps({'cases':cases,'all_pass':all(c['accepted'] for c in cases)}, indent=2, sort_keys=True))
raise SystemExit(0 if all(c['accepted'] for c in cases) else 1)
