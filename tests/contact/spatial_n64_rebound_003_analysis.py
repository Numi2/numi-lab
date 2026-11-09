#!/usr/bin/env python3
"""Qualify hash-bound n64 repeats and conditionally compare with n128."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
old = root / "docs/evidence/midpoint-spatial-cost-dynamic-20261108"
expected = "7f8b311536d8bb7cfd8bf4eb053649da020d0c9a20dc621071284d9e207c2394"
saved = sys.argv
try:
    sys.argv = [str(root / "tests/contact/spatial_n128_retry2_analysis.py")]
    with contextlib.redirect_stdout(io.StringIO()):
        baseline = runpy.run_path(sys.argv[0])
finally:
    sys.argv = saved
read64, compare = baseline["read64"], baseline["compare"]

def read_new(label, h):
    prefix = e / ("rk4-n64-rebound-003-h" + label)
    path = Path(str(prefix) + ".json")
    metadata = json.loads(Path(str(prefix) + "-execution.json").read_text())
    if (metadata["binary_sha256"] != expected or metadata["returncode"] != 0 or
        metadata["stopped_reason"] is not None or
        metadata["output_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest() or
        len(metadata["command"]) != 4 or metadata["command"][1] != "64" or
        float(metadata["command"][2]) != h or float(metadata["command"][3]) != .01):
        raise ValueError("unbound or incomplete n64 run: " + label)
    return read64(path, 64, h)

a = read_new("050", 0.0000005)
b = read_new("025", 0.00000025)
a_old = read64(e / "rk4-n64-h050.json", 64, 0.0000005)
b_old = read64(e / "rk4-n64-parallel-h025.json", 64, 0.00000025)
temporal = compare(a, b)
reference_128_temporal = baseline["t128"]
valid = temporal["passes"] and reference_128_temporal["passes"]
spatial = compare(b, baseline["fine"]) if valid else None
result = {
    "schema": "numi.spatial-n64-rebound.v1",
    "n64_temporal": temporal,
    "n128_temporal": reference_128_temporal,
    "n64_to_n128": spatial,
    "spatial_test_valid": valid,
    "spatial_qualified": bool(valid and spatial["passes"]),
    "historical_coarse_comparison": compare(a_old, a),
    "historical_fine_comparison": compare(b_old, b),
    "historical_coarse_byte_identical": (e / "rk4-n64-h050.json").read_bytes() == (e / "rk4-n64-rebound-003-h050.json").read_bytes(),
    "historical_fine_byte_identical": (e / "rk4-n64-parallel-h025.json").read_bytes() == (e / "rk4-n64-rebound-003-h025.json").read_bytes(),
    "meaning": "These CPU reference checks do not qualify native n64 Matter or physical material calibration. Force peaks are sampled separately from 1 ms bins."
}
print(json.dumps(result, indent=2, sort_keys=True))
