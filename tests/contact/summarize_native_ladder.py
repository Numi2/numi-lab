#!/usr/bin/env python3
"""Summarize every completed preregistered native accuracy attempt."""
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
tags = ['125', '0625', '03125', '015625', '0078125', '00390625']
summary = {}
for n in (2, 4, 8):
    for arm in ('ipc_area_midpoint', 'ipc_area'):
        attempts = []
        for tag in tags:
            name = f'native-n{n}-{arm}-h{tag}'
            tracepath, analysispath = p/(name+'.json'), p/(name+'-analysis.json')
            if not analysispath.exists():
                continue
            trace = json.loads(tracepath.read_text())
            analysis = json.loads(analysispath.read_text())
            attempt = {'h': trace['h'], 'accepted_steps': trace['accepted_steps'],
                'rejected_steps': trace['rejected_steps'], 'passes': analysis['passes'],
                'area_valid': analysis['area_valid'],
                'rejected_state_unchanged': trace['rejected_state_unchanged'],
                'restore_replay': trace['restore_replay'],
                'continuation_replay': trace['continuation_replay']}
            for key in ('binned_force_error', 'peak_force_error', 'total_impulse_error',
                        'cumulative_impulse_error'):
                if key in analysis:
                    attempt[key] = analysis[key]
            if trace['rejected_steps']:
                last = trace['traces'][-1]
                attempt.update({'failure_status': last['status'],
                    'failure_outer_status': last['outer_status'],
                    'restored_state_gap_m': last['gap'], 'failure_diagnostic_sample_contacts': last['contacts'],
                    'failure_diagnostics': last['status_values']})
            attempts.append(attempt)
        summary[f'n{n}-{arm}'] = {
            'attempts': attempts,
            'largest_tested_passing_h': next((a['h'] for a in attempts if a['passes']), None)}
print(json.dumps(summary, indent=2))
