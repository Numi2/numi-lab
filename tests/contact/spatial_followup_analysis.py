#!/usr/bin/env python3
"""Reproduce the preregistered n=16 temporal and n=8-to-16 tests."""
import contextlib
import io
import json
import pathlib
import runpy
import sys

path = pathlib.Path(sys.argv[1])
original = sys.argv
try:
    sys.argv = [str(pathlib.Path(__file__).with_name('spatial_refinement_analysis.py')), str(path)]
    with contextlib.redirect_stdout(io.StringIO()):
        baseline = runpy.run_path(sys.argv[0])
finally:
    sys.argv = original
read, compare = baseline['read'], baseline['compare']
coarse, fine, previous = read(16, '050'), read(16, '025'), read(8, '025')
temporal = compare(coarse, fine)
spatial = compare(previous, fine)
print(json.dumps({
    'n16_temporal': temporal,
    'n8_to_16': spatial,
    'predicted_nonconvergence_confirmed': temporal['passes'] and not spatial['passes'],
    'peak_n16_N': fine['peak_force_N'],
    'impulse_n16_Ns': fine['impulse_Ns'],
    'area_n16_m2': fine['area_m2'],
}, indent=2))
