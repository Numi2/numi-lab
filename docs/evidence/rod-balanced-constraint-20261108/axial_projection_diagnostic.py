#!/usr/bin/env python3
"""Diagnostic scalar FP32 model of the documented XPBD axial projections.

This isolates straight, collinear, hard-endpoint axial dynamics. It is not an
emulation of Metal instruction rounding, bending, contact, or the full solver.
"""
import json
import math
import numpy as np

f = np.float32
rest_total = 0.11758755785232791
radius = 0.000020
density = 1100.0
ea = 2.3818718058345718
h = f(0.001)
rest = f(rest_total / 2)
edge_mass = density * math.pi * radius**2 * (rest_total / 2)
inverse_mass = [f(2 / edge_mass), f(1 / edge_mass), f(2 / edge_mass)]
alpha = f(f(rest / f(ea)) / f(h * h))

def run(start, velocity):
    x = [f(0), f(f(start) + f(h * f(velocity))), f(0.122)]
    predicted = list(x)
    lam = [f(0), f(0)]
    results = []
    for i in range(128):
        old = list(x)
        maximum_projection = 0.0
        for edge in range(2):
            direction = f(1 if x[edge + 1] > x[edge] else -1)
            length = f(abs(f(x[edge + 1] - x[edge])))
            constraint = f(length - rest)
            delta = f(f(-constraint - f(alpha * lam[edge])) /
                      f(f(inverse_mass[edge] + inverse_mass[edge + 1]) + alpha))
            lam[edge] = f(lam[edge] + delta)
            left = f(f(-inverse_mass[edge] * delta) * direction)
            right = f(f(inverse_mass[edge + 1] * delta) * direction)
            x[edge] = f(x[edge] + left)
            x[edge + 1] = f(x[edge + 1] + right)
            maximum_projection = max(maximum_projection, abs(float(left)), abs(float(right)))
        for node, target in ((0, f(0)), (2, f(0.122))):
            correction = f(target - x[node])
            x[node] = f(x[node] + correction)
            maximum_projection = max(maximum_projection, abs(float(correction)))
        movement = max(abs(float(f(a - b))) for a, b in zip(x, old))
        residuals = []
        for edge in range(2):
            length = f(abs(f(x[edge + 1] - x[edge])))
            constraint = f(length - rest)
            residuals.append(float(f(constraint + f(alpha * lam[edge]))))
        results.append(dict(iteration=i + 1, position=list(map(float, x)),
                            net_movement=movement,
                            maximum_single_projection=maximum_projection,
                            stretch_stationarity=residuals,
                            multipliers=list(map(float, lam))))
        if movement <= 1e-7:
            break
    return dict(predicted=list(map(float, predicted)), accepted_by_net_test=movement <= 1e-7,
                final=results[-1], first=results[0])

print(json.dumps(dict(model="collinear FP32 scalar approximation", rest=float(rest),
                      inverse_masses=list(map(float, inverse_mass)), alpha=float(alpha),
                      symmetric=run(.061, 0), asymmetric=run(.060, 0),
                      moving=run(.060, .01)), indent=2, sort_keys=True))
