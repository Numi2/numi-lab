#!/usr/bin/env python3
"""Descriptive summary of already recorded latency-study command-buffer times."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
paths = sorted((ROOT / "docs/evidence/benchmark-latency-results-20261107").glob("seed-*/stdout.json"))
groups = {100: [], 200: []}
for path in paths:
    record = json.loads(path.read_text())
    f = record["raw_fields"]
    n = int(f["requested_steps"])
    if n not in groups or not record["valid"] or int(f["accepted_env_steps"]) != n:
        raise ValueError(f"unexpected or incomplete series: {path}")
    gpu, submission, call, wall = (float(f[k]) for k in ("gpu_ms", "submission_ms", "call_sum_ms", "wall_ms"))
    if not all(math.isfinite(x) and x > 0 for x in (gpu, submission, call, wall)):
        raise ValueError(f"invalid timing: {path}")
    groups[n].append((gpu / n, submission / n, call / n, gpu / wall))
for n, rows in groups.items():
    if len(rows) != 5:
        raise ValueError(f"expected five series of {n} calls, found {len(rows)}")
    med = [statistics.median(values) for values in zip(*rows)]
    print(f"{n} calls: median per-series GPU {med[0]:.6f} ms/call; "
          f"submission {med[1]:.6f} ms/call; API call {med[2]:.6f} ms/call; "
          f"GPU/wall {med[3]:.6f}")
