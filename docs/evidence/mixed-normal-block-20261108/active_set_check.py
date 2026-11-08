#!/usr/bin/env python3
"""FP32 candidate arithmetic against independent FP64 active-set enumeration."""
import json
import numpy as np

def candidate(matrix, rhs):
    h = np.asarray(matrix, dtype=np.float32)
    b = np.asarray(rhs, dtype=np.float32)
    a, c, d, e = h[0, 0], h[0, 1], h[1, 0], h[1, 1]
    det = a * e - c * d
    if not np.isfinite(h).all() or not np.isfinite(b).all() or a <= 0 or e <= 0 or det <= np.float32(1e-7) * a * e:
        return None
    both = np.array([(b[0] * e - c * b[1]) / det, (a * b[1] - d * b[0]) / det], dtype=np.float32)
    if np.isfinite(both).all() and (both >= 0).all():
        return both
    p0 = b[0] / a
    if p0 >= 0 and d * p0 - b[1] >= 0:
        return np.array([p0, 0], dtype=np.float32)
    p1 = b[1] / e
    if p1 >= 0 and c * p1 - b[0] >= 0:
        return np.array([0, p1], dtype=np.float32)
    if (b <= 0).all():
        return np.zeros(2, dtype=np.float32)
    return None

def reference(matrix, rhs):
    h = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    for active in ((0, 1), (0,), (1,), ()):
        p = np.zeros(2)
        if active:
            inds = list(active)
            p[inds] = np.linalg.solve(h[np.ix_(inds, inds)], b[inds])
        gap = h @ p - b
        if np.min(p) >= -1e-12 and np.min(gap) >= -1e-12 and np.max(np.abs(p * gap)) <= 1e-12:
            return p
    raise AssertionError("No feasible FP64 active set")

h = np.array([[4., 1.], [1., 3.]])
cases = [
    ("both", h, h @ [1., 2.]),
    ("first", h, h @ [1., 0.] - [0., 0.7]),
    ("second", h, h @ [0., 2.] - [0.8, 0.]),
    ("neither", h, np.array([-1., -2.])),
]
captured = np.array([[1720.077759, 1712.38623], [1712.386475, 1708.08667]])
old = np.array([3.27708176e-6, 3.491526513e-6])
rhs = np.array([-8.64509e-5, 7.14017e-5])
cases.append(("captured", captured, captured @ old + rhs))
results = []
for name, matrix, b in cases:
    result = candidate(matrix, b)
    expected = reference(matrix, b)
    if result is None or not np.allclose(result, expected, rtol=5e-5, atol=2e-9):
        raise AssertionError((name, result, expected))
    gap = matrix @ result.astype(np.float64) - b
    if (result < 0).any() or np.min(gap) < -2e-6 or np.max(np.abs(result * gap)) > 2e-7:
        raise AssertionError((name, result, gap))
    results.append({"case": name, "candidate": result.tolist(), "reference": expected.tolist(), "gap": gap.tolist()})
for name, matrix in [
    ("near_singular", [[1., 0.99999999], [0.99999999, 1.]]),
    ("indefinite", [[1., 2.], [2., 1.]]),
    ("negative_diagonal", [[-1., 0.], [0., 1.]]),
    ("nonfinite", [[float("nan"), 0.], [0., 1.]]),
]:
    if candidate(matrix, [1., 1.]) is not None:
        raise AssertionError(name)
    results.append({"case": name, "result": "rejected"})
print(json.dumps({"status": "passed", "cases": results}, indent=2))
