#!/usr/bin/env python3
"""Source-bound row-scale audit of the native source380 point trace."""
from __future__ import annotations
import argparse, hashlib, json, math, struct
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def f32(word: str) -> float:
    return struct.unpack('<f', int(word, 16).to_bytes(4, 'little'))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', type=Path, required=True)
    ap.add_argument('--point-input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    d = json.loads(args.trace.read_text())
    inp = json.loads(args.point_input.read_text())
    case = inp['cases'][0]
    if d['point_input_sha256'] != sha(args.point_input):
        raise SystemExit('point input hash does not match the trace header')
    if d['point']['tet_id'] != case['tet_id'] or d['point']['environment'] != case['environment']:
        raise SystemExit('trace and source-bound point identity differ')
    vals = [f32(w) for w in d['trace_f32_bits']]
    accepted = [f32(w) for w in case['accepted_state_f32_bits']]
    floor = f32(case['parameters_f32_bits'][26])
    hdr = int(d['trace_layout']['header_slots'])
    stride = int(d['trace_layout']['iteration_slots'])
    # State count is independently recorded in the fixed trace header.
    n = int(vals[1])
    trace_iterations = int(d['trace_layout']['max_iterations'])
    if (n < 1 or n > 16 or trace_iterations < 1 or
            len(vals) != hdr + trace_iterations * stride):
        raise SystemExit('unexpected native projection trace layout')
    rows = []
    for k in range(int(vals[2])):
        base = hdr + k * stride
        rec = vals[base:base + stride]
        if len(rec) != stride or rec[0] == 0:
            continue
        state = [rec[6+i] + rec[6+16+i] for i in range(n)]
        residual = rec[6+2*16:6+3*16][:n]
        jac_flat = rec[6+5*16:6+5*16+16*16][:n*n]
        jac = [jac_flat[i*n:(i+1)*n] for i in range(n)]
        scales = [max(abs(accepted[i]), abs(state[i]), floor) for i in range(n)]
        row_scales = [sum(abs(jac[i][j]) * scales[j] for j in range(n))
                       for i in range(n)]
        if any(not math.isfinite(value) or value <= 0.0 for value in row_scales):
            raise SystemExit(f'invalid row scales at trace iteration {k+1}')
        raw_index = max(range(n), key=lambda i: abs(residual[i]))
        scaled = [abs(residual[i]) / row_scales[i] for i in range(n)]
        scaled_index = max(range(n), key=lambda i: scaled[i])
        trials = rec[6+4*16:6+5*16][:16]
        rows.append({
            'iteration': k + 1,
            'status': int(rec[0]),
            'backtrack': int(rec[1]),
            'alpha': float(rec[2]),
            'raw_residual_inf': float(rec[3]),
            'scaled_correction_inf': float(rec[5]),
            'raw_residual_dominant_row': raw_index,
            'raw_residual_dominant_value': float(residual[raw_index]),
            'raw_residual_dominant_row_scale': float(row_scales[raw_index]),
            'fixed_row_scaled_residual_inf': float(scaled[scaled_index]),
            'fixed_row_scaled_dominant_row': scaled_index,
            'fixed_row_scales': row_scales,
            'native_raw_trial_residual_inf': [float(x) for x in trials],
            'native_trial_row_vectors_available': False,
            'native_trial_scaled_merits_available': False,
        })
    by = {r['iteration']: r for r in rows}
    evidence = []
    for i in (4, 8, 14):
        r = by[i]
        evidence.append({
            'iteration': i,
            'raw_row': r['raw_residual_dominant_row'],
            'raw_abs': abs(r['raw_residual_dominant_value']),
            'raw_row_scale': r['raw_residual_dominant_row_scale'],
            'scaled_row': r['fixed_row_scaled_dominant_row'],
            'scaled_merit': r['fixed_row_scaled_residual_inf'],
            'raw_trial_inf': r['native_raw_trial_residual_inf'],
        })
    report = {
        'schema': 'numi.cardboard.native-fixed-row-armijo-merit-audit.v1',
        'trace_sha256': sha(args.trace),
        'point_input_sha256': sha(args.point_input),
        'trace_source_manifest_sha256': d['base_snapshot_manifest_sha256'],
        'compiled_local_iterations': d['compiled_local_iterations'],
        'material': d['point']['material_name'],
        'identity': {'environment': d['point']['environment'], 'tet_id': d['point']['tet_id']},
        'state_scale_formula': 'max(abs(accepted_state), abs(current_candidate_high+low), parameter[26] scale floor)',
        'row_scale_formula': 'sum_j(abs(current pre-solve J_ij) * state_scale_j)',
        'scaled_merit_formula': 'max_i(abs(current residual_i) / frozen row_scale_i)',
        'unchanged_gates': {
            'raw_residual_tolerance': '2e-8 * (1 + stateNorm)',
            'scaled_correction_inf_tolerance': '2e-8',
            'armijo_coefficient': '1e-4',
            'backtrack_count': 16,
        },
        'claim_limit': 'This recomputes current-point row scales from native trace words. Original trace388 contains only raw trial max values; it does not prove that any native scaled-merit trial would pass. A follow-up native trace with the overlay is required.',
        'selected_iterations': evidence,
        'all_recorded_iterations': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
