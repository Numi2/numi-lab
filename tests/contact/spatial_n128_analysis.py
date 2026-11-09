#!/usr/bin/env python3
"""Apply preregistered temporal/spatial gates to retained n=64/128 traces.

The n=64 fine trace uses the separately parity-checked ordered evaluator.
A spatial result is meaningful only after both independent temporal tests pass.
"""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[2]
evidence = root / "docs/evidence/midpoint-first-failure-20261109"
argv = sys.argv
try:
    sys.argv = [str(Path(__file__).with_name("spatial_n64_analysis.py")),
                str(evidence / "rk4-n64-parallel-h025.json")]
    with contextlib.redirect_stdout(io.StringIO()):
        previous = runpy.run_path(sys.argv[0])
finally:
    sys.argv = argv
compare = previous["compare"]
read64 = previous["read"]
verify_reference_areas = previous["verify_reference_areas"]

def read128(hcode, h):
    path = evidence / ("rk4-n128-retry1-h" + hcode + ".json")
    execution = json.loads(path.with_name(path.stem + "-execution.json").read_text())
    if execution["returncode"] != 0 or execution["stopped_reason"] is not None:
        raise ValueError("reference did not complete: " + str(path))
    expected = (evidence / "n128-final-inputs.sha256").read_text().splitlines()[1].split()[0]
    if execution["binary_sha256"] != expected:
        raise ValueError("reference binary does not match retained parity record")
    x = json.loads(path.read_text())
    if (x["schema"] != "numi.independent-fem-rk4.v1" or
        x["mesh_divisions"] != 128 or x["h"] != h or x["duration"] != .01 or
        x["nodes"] != 129**3 or x["tetrahedra"] != 6*128**3):
        raise ValueError("incorrect reference configuration: " + str(path))
    weights = x["contact_area_weights_m2"]
    verify_reference_areas(weights,128)
    if (abs(x["area_m2"]-.0004) > 1e-14
        or abs(x["mass_kg"]-.0088) > 1e-8 or x["min_gap_m"] <= 1e-7):
        raise ValueError("invalid area, mass, or clearance: " + str(path))
    if (len(x["trace"]) != 5001 or len(x["force_bins_N"]) != 10 or
        not all(all(math.isfinite(v) for v in row) for row in x["trace"]) or
        not all(math.isfinite(v) and v >= 0 for v in x["force_bins_N"])):
        raise ValueError("invalid trajectory: " + str(path))
    return x, hashlib.sha256(path.read_bytes()).hexdigest()

coarse, coarse_hash = read128("050", .0000005)
fine, fine_hash = read128("025", .00000025)
n64coarse = read64(evidence / "rk4-n64-h050.json",64,.0000005)
n64fine = read64(evidence / "rk4-n64-parallel-h025.json",64,.00000025)
t64, t128 = compare(n64coarse,n64fine), compare(coarse,fine)
spatial = compare(n64fine,fine)
valid = t64["passes"] and t128["passes"]
result = {"n64_temporal":t64, "n128_temporal":t128,
          "n64_to_128":spatial, "spatial_test_valid":valid,
          "spatial_qualified":valid and spatial["passes"],
          "n128_peak_force_N":fine["peak_force_N"],
          "n128_impulse_Ns":fine["impulse_Ns"],
          "n128_reference_area_m2":fine["area_m2"],
          "n128_min_gap_m":fine["min_gap_m"],
          "inputs_sha256":{"h050":coarse_hash,"h025":fine_hash},
          "meaning":"CPU reference for this fixture only; sampled peak and binned force are distinct"}
print(json.dumps(result,indent=2))
