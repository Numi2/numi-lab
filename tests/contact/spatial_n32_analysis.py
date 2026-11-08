#!/usr/bin/env python3
"""Reproduce the preregistered n=32 temporal and n=16-to-32 tests."""
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
coarse, fine, previous = read(32, '050'), read(32, '025'), read(16, '025')
temporal = compare(coarse, fine)
spatial = compare(previous, fine)
previous_temporal = compare(read(16, '050'), previous)
valid = temporal['passes'] and previous_temporal['passes']
print(json.dumps({
    'n32_temporal': temporal,
    'n16_to_32': spatial,
    'n16_temporal': previous_temporal,
    'spatial_test_valid': valid,
    'predicted_nonconvergence_confirmed': valid and not spatial['passes'],
    'peak_n32_N': fine['peak_force_N'],
    'impulse_n32_Ns': fine['impulse_Ns'],
    'area_n32_m2': fine['area_m2'],
}, indent=2))
