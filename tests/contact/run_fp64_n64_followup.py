#!/usr/bin/env python3
"""Run the preregistered n=64 independent references; retain failures."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[2]
evidence = root / "docs/evidence/midpoint-first-failure-20261109"
binary = "/Users/numi/numi-matter-midpoint-reference-build-20261108/numi_contact_fem_fp64_reference_n64"
os.nice(19)
for tag, h in (("050", "0.0000005"), ("025", "0.00000025")):
    output = evidence / f"rk4-n64-h{tag}.json"
    log = evidence / f"rk4-n64-h{tag}.log"
    record = evidence / f"rk4-n64-h{tag}-execution.json"
    if any(x.exists() for x in (output, log, record)):
        raise SystemExit("attempt already exists; inspect it before resuming")
    started = time.monotonic()
    try:
        with output.open("x") as out, log.open("x") as err:
            done = subprocess.run([binary, "64", h, "0.01"], stdout=out,
                                  stderr=err, timeout=7200)
        code, timed_out = done.returncode, False
    except subprocess.TimeoutExpired:
        code, timed_out = None, True
    record.write_text(json.dumps({"mesh": 64, "h": float(h), "returncode": code,
        "timed_out": timed_out, "wall_seconds": time.monotonic()-started}, indent=2)+"\n")
    print("n64", h, "exit", code, "timeout", timed_out, flush=True)
    if code != 0:
        raise SystemExit(1)
