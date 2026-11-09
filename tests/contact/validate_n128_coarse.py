#!/usr/bin/env python3
"""Validate the completed coarse n=128 trace while the fine trace is pending."""
import contextlib,io,json,runpy,sys,math,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2]
e=root/"docs/evidence/midpoint-first-failure-20261109"
saved=sys.argv
try:
 sys.argv=[str(root/"tests/contact/spatial_n64_analysis.py"),str(e/"rk4-n64-parallel-h025.json")]
 with contextlib.redirect_stdout(io.StringIO()): m=runpy.run_path(sys.argv[0])
finally: sys.argv=saved
p=e/"rk4-n128-retry1-h050.json"
x=json.loads(p.read_text())
m["verify_reference_areas"](x["contact_area_weights_m2"],128)
assert x["mesh_divisions"]==128 and x["h"]==5e-7 and x["duration"]==.01
assert len(x["trace"])==5001 and all(all(math.isfinite(v) for v in row) for row in x["trace"])
assert x["min_gap_m"]>1e-7 and abs(x["mass_kg"]-.0088)<1e-8
print(json.dumps({"schema":"numi.spatial-128-coarse-validation.v1","sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"reference_area_m2":x["area_m2"],"minimum_gap_m":x["min_gap_m"],"sampled_peak_force_N":x["peak_force_N"],"total_impulse_Ns":x["impulse_Ns"],"pointwise_area_weights_valid":True,"temporal_qualification":"pending h=0.25 microseconds"},indent=2))
