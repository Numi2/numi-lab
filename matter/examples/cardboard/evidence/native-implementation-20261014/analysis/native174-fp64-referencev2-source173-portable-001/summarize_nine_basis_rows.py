#!/usr/bin/env python3
"""Lossless detail for row/directions without adjacent centered FD passes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def scalar(x):
    x=float(x)
    return x if np.isfinite(x) else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arrays',required=True,type=Path)
    ap.add_argument('--report',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    z=np.load(a.arrays)
    pows=z['epsilon_powers'].tolist()
    records=[]
    for d in range(9):
        misses=np.flatnonzero(~z['adjacent_pair_pass'][d])
        for row in misses:
            entries=[]
            for e,pow2 in enumerate(pows):
                entries.append({
                    'power':int(pow2),
                    'plus_strict':bool(z['valid_plus'][d,e,row]),
                    'minus_strict':bool(z['valid_minus'][d,e,row]),
                    'plus_same_branch':bool(z['same_plus'][d,e,row]),
                    'minus_same_branch':bool(z['same_minus'][d,e,row]),
                    'relative_l2_centered':scalar(z['centered_relative_l2'][d,e,row]),
                    'absolute_l2_centered':scalar(z['centered_absolute_l2'][d,e,row]),
                    'reference_action_l2':scalar(z['reference_action_l2'][d,e,row]),
                    'relative_l2_plus_one_sided':scalar(z['plus_one_sided_relative_l2'][d,e,row]),
                    'relative_l2_minus_one_sided':scalar(z['minus_one_sided_relative_l2'][d,e,row]),
                    'realized_plus_delta':scalar(z['plus_delta'][d,e,row]),
                    'realized_minus_delta':scalar(z['minus_delta'][d,e,row]),
                    'plus_scaled_residual':scalar(z['plus_residual'][d,e,row]),
                    'minus_scaled_residual':scalar(z['minus_residual'][d,e,row]),
                    'plus_dynamic_correction':scalar(z['plus_correction'][d,e,row]),
                    'minus_dynamic_correction':scalar(z['minus_correction'][d,e,row]),
                    'plus_iterations':int(z['plus_iterations'][d,e,row]),
                    'minus_iterations':int(z['minus_iterations'][d,e,row]),
                    'plus_failure_reason':str(z['plus_reason'][d,e,row]),
                    'minus_failure_reason':str(z['minus_reason'][d,e,row]),
                })
            row_passes=int(z['centered_pass'][d,:,row].sum())
            records.append({
                'direction_index':d,
                'F_component':[d//3,d%3],
                'paper_row':int(row),
                'tet':int(z['paper_tets'][row]),
                'grid_centered_pass_count':row_passes,
                'adjacent_pass':False,
                'classification':'sampled-grid qualification unresolved; all comparisons remain reported and no absolute noise floor is established',
                'epsilon_results':entries,
            })
    summ=json.load(a.report.open())
    out={
        'schema':'numi.cardboard.bilinear15.nine-basis-fd-row-detail.v1',
        'source_report_sha256':sha(a.report),
        'arrays_sha256':sha(a.arrays),
        'epsilon_powers':pows,
        'row_direction_count':9*len(z['paper_tets']),
        'adjacent_centered_pass_count':sum(x['rows_with_adjacent_centered_passes'] for x in summ['per_basis_direction']),
        'row_directions_without_adjacent_pass':len(records),
        'row_direction_details':records,
        'interpretation':'A missing adjacent pair is not silently converted to pass. All reported cases have strict endpoint passes, center-branch agreement, nonzero realized deltas, and nonzero reference actions in the parent audit; the finite-grid result still lacks an absolute noise-floor measurement.',
    }
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'row_directions_without_adjacent_pass':len(records),'output_sha256':sha(a.output)}))
if __name__=='__main__': main()
