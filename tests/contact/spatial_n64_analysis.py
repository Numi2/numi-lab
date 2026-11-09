#!/usr/bin/env python3
"""Apply the original reference gates to n=32 and independently checked n=64."""
import contextlib
import io
import json
import math
from pathlib import Path
import runpy
import sys
root=Path(__file__).resolve().parents[2]
old=root/"docs/evidence/midpoint-spatial-cost-dynamic-20261108"
new=root/"docs/evidence/midpoint-first-failure-20261109"
argv=sys.argv
try:
    sys.argv=[str(Path(__file__).with_name("spatial_refinement_analysis.py")),str(old)]
    with contextlib.redirect_stdout(io.StringIO()):
        baseline=runpy.run_path(sys.argv[0])
finally:
    sys.argv=argv
compare=baseline["compare"]
verify_reference_areas=baseline["verify_reference_areas"]
def read(path,n,h):
    x=json.loads(path.read_text())
    assert x["mesh_divisions"]==n and x["duration"]==.01 and x["h"]==h
    assert len(x["trace"])==5001 and len(x["force_bins_N"])==10
    verify_reference_areas(x["contact_area_weights_m2"],n)
    assert abs(x["area_m2"]-.0004)<1e-14 and x["min_gap_m"]>1e-7
    assert all(all(math.isfinite(v) for v in row) for row in x["trace"])
    return x
old_coarse=read(old/"rk4-n32-h050.json",32,.0000005)
old_fine=read(old/"rk4-n32-h025.json",32,.00000025)
new_coarse=read(new/"rk4-n64-h050.json",64,.0000005)
new_fine_path=Path(sys.argv[1]) if len(sys.argv)==2 else new/"rk4-n64-h025.json"
new_fine=read(new_fine_path,64,.00000025)
t32=compare(old_coarse,old_fine); t64=compare(new_coarse,new_fine)
spatial=compare(old_fine,new_fine); valid=t32["passes"] and t64["passes"]
print(json.dumps({"n32_temporal":t32,"n64_temporal":t64,"n32_to_64":spatial,
 "spatial_test_valid":valid,"spatial_qualified":valid and spatial["passes"],
 "n64_peak_force_N":new_fine["peak_force_N"],"n64_impulse_Ns":new_fine["impulse_Ns"],
 "n64_reference_area_m2":new_fine["area_m2"],"n64_min_gap_m":new_fine["min_gap_m"],
 "n64_fine_file":str(new_fine_path)},indent=2))
