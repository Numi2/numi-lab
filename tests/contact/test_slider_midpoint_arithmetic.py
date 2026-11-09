#!/usr/bin/env python3
"""CPU-only equation check; this does not qualify a native transaction."""
from decimal import Decimal, getcontext
from pathlib import Path
import math
import random
import struct
import subprocess
import tempfile

getcontext().prec = 100
ROOT = Path(__file__).resolve().parents[2]

def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]

def value(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]

def dec(word):
    return Decimal.from_float(value(word))

def pair(x, rng):
    hi = bits(x)
    # A low term much smaller than an ulp, with both signs represented.
    low = bits(math.ldexp(value(hi), -28) * rng.uniform(-0.9, 0.9))
    return hi, low

rng = random.Random(0x71D0)
cases = []
# The accepted state immediately preceding the retained rejected trajectory.
# The tentative increment is an arithmetic input, not a reproduced solve.
cases.append((bits(-0.00024366525758523494), bits(0.0),
              bits(-0.07551109045743942), bits(0.0),
              bits(-9.81 * 1.5625000742147677e-5 / 2), bits(0.0),
              bits(1.5625000742147677e-5)))
for _ in range(10000):
    q = pair(rng.uniform(-1., 1.) * 10**rng.uniform(-5, -1), rng)
    v = pair(rng.uniform(-1., 1.) * 10**rng.uniform(-3, 0), rng)
    dv = pair(rng.uniform(-1., 1.) * 10**rng.uniform(-6, -2), rng)
    h = bits(10**rng.uniform(-7, -3))
    cases.append((*q, *v, *dv, h))
with tempfile.TemporaryDirectory(prefix="slider-arithmetic-") as directory:
    executable = Path(directory) / "check"
    subprocess.run(["clang++", "-std=c++20", "-O2", "-fno-fast-math",
                    "-ffp-contract=off", "-I", str(ROOT / "include"),
                    str(ROOT / "tests/contact/slider_midpoint_arithmetic_check.cpp"),
                    "-o", str(executable)], check=True)
    inputs = "".join(" ".join(format(x, "x") for x in case) + "\n" for case in cases)
    proc = subprocess.run([str(executable)], input=inputs, text=True,
                          capture_output=True, check=True)
lines = proc.stdout.splitlines()
assert len(lines) == len(cases)
worst = Decimal(0)
for case, line in zip(cases, lines):
    words = [int(x, 16) for x in line.split()]
    assert len(words) == 10 and words[8:] == [1, 1], (case, words)
    q = dec(case[0]) + dec(case[1])
    v = dec(case[2]) + dec(case[3])
    dv = dec(case[4]) + dec(case[5])
    h = dec(case[6])
    expected = (q + h * (v + dv) / 2, v + dv,
                q + h * (v + dv), v + 2 * dv)
    for i, exact in enumerate(expected):
        got = dec(words[2*i]) + dec(words[2*i+1])
        # Scale by terms, rather than the potentially cancelled result.
        scale = (abs(q) + abs(h*(v+dv)) if i % 2 == 0
                 else abs(v) + 2*abs(dv))
        ratio = abs(got-exact) / max(scale, Decimal("1e-35"))
        worst = max(worst, ratio)
        assert ratio < Decimal(2)**-43, (case, i, str(ratio))
print(f"{len(cases)} exact-decimal equation cases passed; worst scaled error {worst:.3E}")
print("CPU arithmetic only: no Metal execution, accepted state, or rollback is qualified")
