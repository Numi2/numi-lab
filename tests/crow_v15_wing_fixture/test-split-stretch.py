#!/usr/bin/env python3
"""Native Metal arithmetic reference; does not exercise world integration."""
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile

if len(sys.argv) != 3:
    raise SystemExit('usage: test-split-stretch.py DRIVER SHADER')
driver, shader = map(Path, sys.argv[1:])
rng = random.Random(506311)
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
rest = .003125
ea = 2.129e9 * 3.449e-6
inputs, reference = bytearray(), []
for shift in (0., 1., 16.):
    for _ in range(32):
        a = [shift + rng.uniform(-.01, .01), rng.uniform(-.01, .01), rng.uniform(-.01, .01)]
        direction = [rng.gauss(0, 1) for _ in range(3)]
        norm = math.hypot(*direction)
        length = rng.uniform(.003120, .003130)
        b = [a[k] + length * direction[k] / norm for k in range(3)]
        ah, bh = [f32(x) for x in a], [f32(x) for x in b]
        al = [f32(a[k] - ah[k]) for k in range(3)]
        bl = [f32(b[k] - bh[k]) for k in range(3)]
        rh = f32(rest)
        inputs.extend(struct.pack('<20f', *(ah + [0] + al + [0] + bh + [0] + bl + [0] + [rh, f32(rest-rh), f32(ea), 0])))
        reference.append(ea * (math.dist(a, b) - rest) / rest)
with tempfile.TemporaryDirectory(prefix='crow-split-stretch-') as tmp:
    src, out = Path(tmp) / 'input.bin', Path(tmp) / 'output.bin'
    src.write_bytes(inputs)
    run = subprocess.run([str(driver), str(shader), str(src), str(out)], capture_output=True, text=True)
    if run.returncode:
        raise AssertionError(f'native driver failed {run.returncode}: {run.stderr}')
    metadata = json.loads(run.stdout)
    assert metadata['fast_math'] is False and metadata['cases'] == len(reference)
    data = out.read_bytes()
    assert len(data) == len(reference) * 16
    errors, controls = [], []
    for i, expected in enumerate(reference):
        actual, naive, hi, lo = struct.unpack_from('<4f', data, i * 16)
        assert all(math.isfinite(x) for x in (actual, naive, hi, lo, expected))
        errors.append(abs(actual - expected))
        controls.append(abs(naive - expected))
    assert max(errors) <= .0001, max(errors)
    assert max(controls) > .01, max(controls)
    print(f"PASS: {len(reference)} cases, max error {max(errors):.9g} N, max naive error {max(controls):.9g} N")
