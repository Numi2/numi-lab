"""Exact per-arm first-terminal and selected-environment trace parity."""
import json
from pathlib import Path

CONTRACT = {'world_fingerprint':'3978934575658225183', 'task_fingerprint':'922021427268694894',
            'observation_fingerprint':'15900562835394482329', 'action_fingerprint':'9497262916478774679'}

def compare(root, first, second, full_trace):
    root = Path(root)
    a=json.loads((root/(first+'.stdout')).read_text())
    b=json.loads((root/(second+'.stdout')).read_text())
    for x in (a,b):
        if x['failed_environment_steps'] != 0: raise ValueError('physics failure')
        for k, v in CONTRACT.items():
            if str(x[k]) != v: raise ValueError('contract mismatch '+k)
    keys=['failed_environment_steps','first_termination_step_by_environment','first_termination_tilt_rad_by_environment']
    differences={k:[a[k],b[k]] for k in keys if a[k] != b[k]}
    arows=(root/(first+'.trace.tsv')).read_text().splitlines()
    brows=(root/(second+'.trace.tsv')).read_text().splitlines()
    if len(arows)!=1001 or len(brows)!=1001: raise ValueError('trace length')
    step=a['first_termination_step_by_environment'][0]
    if not isinstance(step,int) or not 1<=step<=1000: raise ValueError('terminal step')
    end=1001 if full_trace else 1+step
    mismatches=[i for i in range(end) if arows[i] != brows[i]]
    return {'compared_trace_lines':end,'trace_mismatch_count':len(mismatches),
            'first_mismatch_line':mismatches[0] if mismatches else None,'differences':differences}
