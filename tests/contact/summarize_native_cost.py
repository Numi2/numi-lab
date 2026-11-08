#!/usr/bin/env python3
"""Summarize complete snapshot-free rollouts bound to an accuracy trace."""
import json
import statistics
import sys

with open(sys.argv[1]) as f:
    reference = json.load(f)
runs = []
for path in sys.argv[2:]:
    with open(path) as f:
        a = json.load(f)
    assert a['h'] == reference['h']
    assert a['divisions'] == reference['divisions']
    assert a['arm'] == reference['arm']
    assert a['requested_steps'] == reference['accepted_steps']
    assert a['accepted_steps'] == a['requested_steps']
    assert a['rejected_steps'] == 0 and a['status'] == 0
    assert a['per_step_snapshots'] == 0 and a['capture_diagnostics']
    assert a['source_fingerprint'] == reference['source_fingerprint']
    assert a['device_fingerprint'] == reference['device_fingerprint']
    assert a['final_physical_sha256'] == reference['final_physical_sha256']
    runs.append(a)
assert len(runs) == 3
print(json.dumps({
    'arm': reference['arm'], 'divisions': reference['divisions'],
    'timestep_seconds': reference['h'],
    'wall_seconds': [a['wall_seconds'] for a in runs],
    'median_wall_seconds': statistics.median(a['wall_seconds'] for a in runs),
    'accepted_seconds_per_wall_second': [a['accepted_seconds_per_wall_second'] for a in runs],
    'median_accepted_seconds_per_wall_second': statistics.median(a['accepted_seconds_per_wall_second'] for a in runs),
    'all_final_digests_match_qualified_trace': True,
    'accuracy_qualification_must_be_checked_separately': True,
}, indent=2))
