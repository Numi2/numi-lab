#!/usr/bin/env python3
"""Check MetalWorld normal complementarity reporting at several mass scales."""
import json
import math
import subprocess
import sys


def residual(case, row):
    if (row['bodyA'], row['bodyB']) != (0, 1):
        raise AssertionError('fixture did not produce the expected plane-box contact')
    v = case['v']
    x, y, z = row['point']
    rx, ry, rz = x, y - case['initial_y'], z
    point_v = (v[0] + v[4]*rz - v[5]*ry,
               v[1] + v[5]*rx - v[3]*rz,
               v[2] + v[3]*ry - v[4]*rx)
    equation = sum(a*b for a, b in zip(point_v, row['direction']))
    equation -= row['target']
    equation += row['reg']*row['lambda']
    return abs(equation) if row['lambda'] > 0.0 else max(-equation, 0.0)


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: test.py PROBE METALLIB')
    completed = subprocess.run(sys.argv[1:], check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cases = json.loads(completed.stdout)['cases']
    if len(cases) != 36:
        raise AssertionError(f'expected 36 cases, got {len(cases)}')
    positive_small = 0
    nonzero_residual = 0
    for case in cases:
        if not case['success'] or not case['repeat'] or case['contacts'] != 2:
            raise AssertionError(f'fixture failed: {case}')
        independent = max(residual(case, row) for row in case['rows'])
        reported = case['solver_residual']
        if not math.isfinite(independent) or not math.isfinite(reported):
            raise AssertionError('nonfinite residual')
        if abs(independent - reported) > 2e-5:
            raise AssertionError(f'residual mismatch: {case}; independently reconstructed={independent}')
        if any(0 < row['lambda'] <= 1e-7 for row in case['rows']):
            positive_small += 1
            nonzero_residual += independent > 0.002
    if positive_small == 0 or nonzero_residual == 0:
        raise AssertionError('fixture did not exercise small positive impulses with a nonzero residual')
    print(f'36 cases passed; {positive_small} include small positive impulses')


if __name__ == '__main__':
    main()
