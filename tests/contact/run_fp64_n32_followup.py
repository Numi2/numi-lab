#!/usr/bin/env python3
"""Run preregistered n=32 references, fail closed and retain every attempt."""
import json
import os
import pathlib
import subprocess
import time

root = pathlib.Path(__file__).resolve().parents[2]
p = root / 'docs/evidence/midpoint-spatial-cost-dynamic-20261108'
b = '/Users/numi/numi-matter-midpoint-reference-build-20261108/numi_contact_fem_fp64_reference_n32'
os.nice(19)
for tag, h in [('050', '0.0000005'), ('025', '0.00000025')]:
    out = p / f'rk4-n32-h{tag}.json'
    log = p / f'rk4-n32-h{tag}.log'
    record = p / f'rk4-n32-h{tag}-execution.json'
    if out.exists() or log.exists() or record.exists():
        raise RuntimeError('attempt exists; inspect before recovery')
    start = time.monotonic()
    try:
        with out.open('x') as stream, log.open('x') as errors:
            result = subprocess.run([b, '32', h, '0.01'], stdout=stream,
                                    stderr=errors, timeout=1800)
        code, timeout = result.returncode, False
    except subprocess.TimeoutExpired:
        code, timeout = None, True
    record.write_text(json.dumps({'mesh': 32, 'h': float(h), 'returncode': code,
        'timed_out': timeout, 'wall_seconds': time.monotonic()-start}, indent=2)+'\n')
    print('n32', h, 'exit', code, 'timeout', timeout, flush=True)
    if code != 0:
        raise SystemExit(1)
