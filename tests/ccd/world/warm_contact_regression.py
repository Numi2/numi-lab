#!/usr/bin/env python3
"""Narrow wave-path containment regression; not a contact-accuracy certificate."""
import json, math, subprocess, sys
exe, shader = sys.argv[1:]
r = subprocess.run([exe, shader], text=True, capture_output=True, check=True)
d = json.loads(r.stdout)
a = d['cases']
assert len(a) == 5
for c in a:
    assert all(s['replay'] for s in c['steps'])
for c in a[:2]:
    assert c['steps'][0]['accepted']
    assert all(not s['accepted'] and s['status'] == 11 and s['restored'] for s in c['steps'][1:])
for s in a[2]['steps']:
    assert s['accepted'] and s['normal_impulse'] == 0 and abs(s['v'][0] - 1) < 1e-6 and abs(s['v'][1]) < 1e-6
for c in a[3:]:
    assert all(s['accepted'] for s in c['steps'])
    assert all(math.isfinite(s['normal_impulse']) and abs(s['normal_impulse'] - .0981) < .0002 for s in c['steps'])
    assert all(s['linear_error'] < 2e-6 and s['angular_error'] < 2e-6 for s in c['steps'])
print('warm-row rejection, full-cache rollback, replay, near miss and two supported contact controls passed')
