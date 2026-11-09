#!/usr/bin/env python3
"""Exercise the actual retained weight verifier against sum-preserving faults."""
import contextlib
import io
import json
import math
from pathlib import Path
import runpy
import sys

here = Path(__file__).resolve().parent
root = here.parents[1]
sys.argv = [str(here / "spatial_refinement_analysis.py"),
            str(root / "docs/evidence/midpoint-spatial-cost-dynamic-20261108")]
with contextlib.redirect_stdout(io.StringIO()):
    verify = runpy.run_path(sys.argv[0])["verify_reference_areas"]
x = json.loads((root / "docs/evidence/midpoint-first-failure-20261109/rk4-n64-h050.json").read_text())
w = x["contact_area_weights_m2"]
verify(w,64)
for name in ("surface redistribution", "wrong layer", "nonfinite"):
    bad = w.copy()
    if name == "surface redistribution":
        value = bad[0]; bad[0] += value; bad[1] -= value
    elif name == "wrong layer":
        value = bad[0]; bad[0] = 0; bad[65*65] = value
    else:
        bad[0] = float("nan")
    if name != "nonfinite":
        assert math.isclose(math.fsum(bad),math.fsum(w),rel_tol=0,abs_tol=1e-19)
    try: verify(bad,64)
    except AssertionError: pass
    else: raise AssertionError(name + " was admitted")
print("reference-area verifier accepts retained n64 weights and rejects redistributed, misplaced and nonfinite weights")
