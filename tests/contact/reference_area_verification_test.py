#!/usr/bin/env python3
"""Exercise the actual retained weight verifier against sum-preserving faults."""
import contextlib
import io
import json
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
for name, mutate in (
    ("surface redistribution", lambda a: (a.__setitem__(0, a[0]*2), a.__setitem__(1, a[1]-a[0]/2))),
    ("wrong layer", lambda a: (a.__setitem__(0,0), a.__setitem__(65*65,a[0]))),
    ("nonfinite", lambda a: a.__setitem__(0,float("nan"))),
):
    bad = w.copy()
    if name == "wrong layer":
        value = bad[0]; bad[0] = 0; bad[65*65] = value
    else:
        mutate(bad)
    try: verify(bad,64)
    except AssertionError: pass
    else: raise AssertionError(name + " was admitted")
print("reference-area verifier accepts retained n64 weights and rejects redistributed, misplaced and nonfinite weights")
