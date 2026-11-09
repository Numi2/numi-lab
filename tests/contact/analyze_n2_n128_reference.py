#!/usr/bin/env python3
"""Compare the timing fixture with the finer, not spatially converged reference."""
import hashlib
import json
from pathlib import Path
from large_spatial_reference import read_reference, compare
root = Path(__file__).resolve().parents[2]
e = root / "docs/evidence/midpoint-first-failure-20261109"
old = root / "docs/evidence/midpoint-spatial-cost-dynamic-20261108"
paths = [old / "rk4-n2-h050.json", old / "rk4-n2-h025.json",
         e / "rk4-n128-retry2-h025.json", e / "spatial-n128-retry2-analysis.json"]
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()
coarse = read_reference(paths[0],2,5e-7)
fine = read_reference(paths[1],2,2.5e-7)
temporal = compare(coarse,fine)
record = json.loads(paths[3].read_text())
if not temporal["passes_refinement"] or not record["n128_temporal"]["passes"] or record["inputs_sha256"]["h025"] != digest(paths[2]):
    raise ValueError("time refinement or n128 reference identity failed")
finer = read_reference(paths[2],128,2.5e-7)
print(json.dumps({"scope":"non-adjacent numerical mesh comparison; n128 is not spatially converged or a calibrated physical reference",
                  "inputs_sha256":{str(p):digest(p) for p in paths},
                  "n2_temporal":temporal,"n128_temporal":record["n128_temporal"],
                  "n2_to_n128":compare(fine,finer),
                  "n2_peak_force_N":fine["peak_force_N"],"n128_peak_force_N":finer["peak_force_N"],
                  "n2_total_impulse_Ns":fine["impulse_Ns"],"n128_total_impulse_Ns":finer["impulse_Ns"]},indent=2,sort_keys=True))
