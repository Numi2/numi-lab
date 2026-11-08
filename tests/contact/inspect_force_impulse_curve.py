#!/usr/bin/env python3
"""Report sampled peak timing and cumulative impulse differences."""
import json
import sys

with open(sys.argv[1]) as f:
    a = json.load(f)
with open(sys.argv[2]) as f:
    reference = json.load(f)
assert a['divisions'] == reference['mesh_divisions'] and not a['rejected_steps']
h = a['h']
rows = a['traces']
assert len(rows) == a['accepted_steps']
cumulative = [0.0]
for i, row in enumerate(rows):
    assert row['accepted'] and row['step'] == i
    cumulative.append(cumulative[-1] + row['accepted_normal_impulse'])
def impulse(t):
    k = min(int(t / h), len(rows) - 1)
    fraction = max(0., min(1., (t - k*h) / h))
    return cumulative[k] + fraction * (cumulative[k+1] - cumulative[k])
peak = max(rows, key=lambda r: r['force'])
peak_time = (peak['step'] + (0.5 if a['arm'].endswith('_midpoint') else 1.0))*h
refpeak = max(reference['trace'], key=lambda r: r[1])
errors = [(t, impulse(t) - val) for t, _, val in reference['trace']]
error_time, diff = max(errors, key=lambda p: abs(p[1]))
samples = []
for t, _, value in reference['trace']:
    if any(abs(t - target) < 1e-12 for target in (0.002, 0.004, 0.006, 0.008, 0.010)):
        samples.append({'t_s': t, 'native_Ns': impulse(t), 'reference_Ns': value})
print(json.dumps({'arm': a['arm'], 'h': h, 'native_sampled_peak_N': peak['force'],
    'native_peak_sample_time_s': peak_time,
    'reference_2us_sampled_peak_N': refpeak[1], 'reference_2us_peak_time_s': refpeak[0],
    'reference_full_025us_sampled_peak_N': reference['peak_force_N'],
    'max_cumulative_error_time_s': error_time, 'max_cumulative_signed_error_Ns': diff,
    'reference_final_impulse_Ns': reference['impulse_Ns'], 'impulse_samples': samples}, indent=2))
