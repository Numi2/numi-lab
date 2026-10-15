"""Small stdlib CPU reference for the opt-in material Armijo merit."""
from __future__ import annotations

import math
from typing import Sequence


def frozen_row_scales(jacobian: Sequence[Sequence[float]],
                      state_scales: Sequence[float]) -> list[float]:
    n = len(state_scales)
    if n == 0 or len(jacobian) != n or any(len(row) != n for row in jacobian):
        raise ValueError("square Jacobian and matching state scales are required")
    if (any(not math.isfinite(float(s)) or float(s) <= 0.0 for s in state_scales) or
            any(not math.isfinite(float(x)) for row in jacobian for x in row)):
        raise ValueError("finite Jacobian and positive finite state scales are required")
    rows = [sum(abs(float(jacobian[i][j])) * float(state_scales[j])
                 for j in range(n)) for i in range(n)]
    if any(not math.isfinite(value) or value <= 0.0 for value in rows):
        raise ValueError("every residual row must have a positive finite scale")
    return rows


def merit(residual: Sequence[float], row_scales: Sequence[float]) -> float:
    if len(residual) != len(row_scales) or not residual:
        return math.inf
    if any(not math.isfinite(float(x)) for x in residual):
        return math.inf
    if any(not math.isfinite(float(s)) or float(s) <= 0.0 for s in row_scales):
        return math.inf
    return max(abs(float(r)) / float(s) for r, s in zip(residual, row_scales))


def armijo_accept(current: float, trial: float, step: float,
                  coefficient: float = 1.0e-4) -> bool:
    if not (math.isfinite(current) and math.isfinite(trial) and
            math.isfinite(step) and 0.0 < step <= 1.0 and
            current >= 0.0 and trial >= 0.0):
        return False
    return trial < current and trial <= (1.0 - coefficient * step) * current
