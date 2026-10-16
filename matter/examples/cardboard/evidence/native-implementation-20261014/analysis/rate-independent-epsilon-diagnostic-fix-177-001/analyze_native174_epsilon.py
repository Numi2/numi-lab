"""Independent endpoint reference and error decomposition; no GPU work."""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json
import sys
import warnings
import numpy as np

B = Path('/Users/home/cardboard-implementation-20261014')
OUT = B / 'analysis/native174-realized-epsilon-reference-003'
RUN = B / 'analysis/native469-material-replay-174-host-macmini-002/runs/invocations/20261016T040030Z-f41839901e414db49c077be20a95edf6/native-output'
ARR = B / 'analysis/native469-bilinear-layer-replay-002/arrays.npz'
MIX = B / 'analysis/native469-mp-fallback-005/run-002/mixed_reference_arrays.npz'
REF = B / 'analysis/bilinear-capture-layer-audit-source166-003/candidate/matter/tools/cardboard_bilinear_fp64_reference.py'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def records(name, kind):
    with (RUN / name).open() as f:
        return [x for line in f if (x := json.loads(line)).get('record') == kind]

def floats(bits):
    return np.array([int(x, 16) for x in bits], dtype=np.uint32).view(np.float32).astype(np.float64)

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(x) for x in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value

def main():
    OUT.mkdir()
    spec = importlib.util.spec_from_file_location('epsilon_independent_ref', REF)
    ref = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ref; spec.loader.exec_module(ref)
    a = np.load(ARR); mix = np.load(MIX)
    paper_tets = np.flatnonzero(a['material'] != 2)
    paper_row = {int(t): i for i, t in enumerate(paper_tets)}
    sample = [x for x in records('epsilon-grid-inputs.jsonl', 'sample')
              if x['column'] == -1 and x['material_index'] in (0, 1)]
    assert len(sample) == 576
    key = lambda x: (x['tet_id'], x['state_index'], x['epsilon'], x['sign'])
    sample_index = {key(x): i for i, x in enumerate(sample)}
    assert len(sample_index) == len(sample)
    inputs = np.array([floats(x['input_f32_bits']) for x in sample])
    old = np.array([floats(x['accepted_state_f32_bits']) for x in sample])
    F = (inputs[:, :9] + inputs[:, 9:18]).reshape(-1, 3, 3)
    centers = {(x['tet_id'], x['state_index']): floats(x['input_f32_bits']) for x in records('all-captured-center-side-inputs.jsonl', 'sample')}
    p = np.array([mix['paper_parameters_f32'][paper_row[x['tet_id']]] for x in sample], dtype=np.float64)
    for i, x in enumerate(sample):
        tet = x['tet_id']; row = paper_row[tet]
        assert np.array_equal(old[i], a['accepted_paper_history'][row])
        assert a['material'][tet] == x['material_index']
        expected_delta = x['sign'] * x['epsilon'] * inputs[i, 45] * inputs[i, 27:36]
        assert np.allclose(expected_delta, x['requested_F_delta'], rtol=1e-8, atol=1e-18)
        center_input = centers[(tet, x['state_index'])]
        center = center_input[:9] + center_input[9:18]
        assert np.array_equal(center, a['s' + str(x['state_index']) + '_F'][tet].reshape(9))
        delta = F[i].reshape(9) - center
        # JSON decimal delta fields are printed at nine significant digits;
        # they must not replace the exact captured input words in the oracle.
        assert np.allclose(delta, x['realized_F_delta'], rtol=1e-8, atol=1e-18)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        base = ref.project_centers(ref, p, F, old, p[:, 26])
        z, it, rn, cn, ok, reasons, passes = ref.continue_projection_tight(ref, p, F, old, p[:, 26], base)
    P = ref.stress(p, F, z)
    branch = ref.paper_branch(ref, p, old, z)
    output = []
    for pair in records('epsilon-fd-comparison.jsonl', 'pair'):
        keys = [(pair['tet_id'], pair['state_index'], pair['epsilon'], sign) for sign in (1, -1)]
        if keys[0] not in sample_index: continue
        ip, im = [sample_index[k] for k in keys]
        tet = pair['tet_id']; epsilon = pair['epsilon']; state = pair['state_index']
        native_p = floats(pair['native_output_bits_plus'])[8:17]
        native_m = floats(pair['native_output_bits_minus'])[8:17]
        prefix = 'center' if state == 0 else 's' + str(state)
        C = mix[prefix + '_tangent'][tet]
        dxp = F[ip].reshape(9) - a['s' + str(state) + '_F'][tet].reshape(9)
        dxm = F[im].reshape(9) - a['s' + str(state) + '_F'][tet].reshape(9)
        prediction = C @ (0.5 * (dxp - dxm))
        cpu_secant = 0.5 * (P[ip].reshape(9) - P[im].reshape(9))
        native_secant = 0.5 * (native_p - native_m)
        truncation = cpu_secant - prediction
        evaluation = native_secant - cpu_secant
        total = native_secant - prediction
        assert np.allclose(total, truncation + evaluation, rtol=1e-10, atol=1e-12)
        scale = np.linalg.norm(prediction)
        # The old diagnostic averaged deltaF with dt*deltaFdot. Preserve and
        # quantify that observation error rather than treating it as physics.
        rate_p = np.array(sample[ip]['realized_rate_delta'])
        rate_m = np.array(sample[im]['realized_rate_delta'])
        prior_delta = 0.25 * ((dxp - dxm) + inputs[ip, 45] * (rate_p - rate_m))
        prior_prediction = C @ prior_delta
        rec = {
            'tet': tet, 'state_index': state, 'epsilon': epsilon, 'endpoint_indices': [ip, im],
            'strict_fp64_endpoints_pass': bool(ok[ip] and ok[im]),
            'endpoint_residuals': rn[[ip, im]], 'endpoint_dynamic_corrections': cn[[ip, im]],
            'endpoint_failure_reasons': [reasons[ip], reasons[im]],
            'center_tangent_reference': 'independent_MP_fallback' if 'mp_tangent_' + str(state) + '_' + str(tet) in mix else 'independent_FP64',
            'realized_F_prediction_norm_Pa': scale,
            'FP64_secant_truncation_absolute_Pa': np.linalg.norm(truncation),
            'native_endpoint_evaluation_contribution_absolute_Pa': np.linalg.norm(evaluation),
            'native_secant_total_error_absolute_Pa': np.linalg.norm(total),
            'FP64_secant_relative_error': np.linalg.norm(truncation) / scale if scale else None,
            'native_secant_relative_error': np.linalg.norm(total) / scale if scale else None,
            'old_rate_averaging_prediction_error_absolute_Pa': np.linalg.norm(prior_prediction - prediction),
            'native_stress_absolute_errors_Pa': [np.linalg.norm(native_p - P[ip].reshape(9)), np.linalg.norm(native_m - P[im].reshape(9))],
            'native_stress_relative_errors': [np.linalg.norm(native_p - P[ip].reshape(9)) / np.linalg.norm(P[ip]), np.linalg.norm(native_m - P[im].reshape(9)) / np.linalg.norm(P[im])],
            'endpoint_branch': {k: branch[k][[ip, im]] for k in branch},
        }
        output.append(clean(rec))
    assert len(output) == 288
    with (OUT / 'pair-comparisons.jsonl').open('x') as f:
        for rec in output: f.write(json.dumps(rec, allow_nan=False) + '\n')
    np.savez_compressed(OUT / 'endpoint-reference-arrays.npz', F=F, parameters=p, accepted=old,
                        projected=z, stress=P, strict_pass=ok, iterations=it, residual=rn,
                        correction=cn, **branch)
    groups = collections.defaultdict(list)
    for rec in output: groups[rec['epsilon']].append(rec)
    table = []
    for eps, rows in sorted(groups.items()):
        healthy = [x for x in rows if x['strict_fp64_endpoints_pass']]
        table.append({'epsilon': eps, 'pairs': len(rows), 'strict_reference_pairs': len(healthy),
                      **{key + '_max': max((x[key] for x in healthy), default=None) for key in (
                          'FP64_secant_relative_error', 'native_secant_relative_error',
                          'native_endpoint_evaluation_contribution_absolute_Pa',
                          'old_rate_averaging_prediction_error_absolute_Pa')}})
    report = {'schema': 'numi.cardboard.native174-realized-epsilon-reference.v1',
              'gpu_executed': False, 'paper_endpoint_count': len(sample),
              'strict_reference_passes': int(ok.sum()), 'strict_reference_failures': int((~ok).sum()),
              'unchanged_reference_gates': {'scaled_residual': 2e-12, 'dynamic_correction': 2e-11},
              'pairs': len(output), 'epsilon_table': table,
              'warnings': sorted(set(str(x.message) for x in caught)),
              'verdict': 'Per-point error decomposition only; no unmeasured noise floor or full material qualification.',
              'limits': ['The native output is the already closed174 replay; no native rerun occurred.',
                         'This law is rate independent. Correct prediction uses realized deltaF only.',
                         'Failed strict FP64 endpoints remain excluded from passing comparisons.',
                         'Center tangents retain their distinct FP64 and MP-fallback provenance.',
                         'Branch indicators are retained; branch-crossing secants are not smooth derivative evidence.',
                         'Endpoint native-reference errors are measured accuracy errors at these inputs, not a universal stochastic noise floor.'],
              'input_sha256': {str(q): sha(q) for q in (ARR, MIX, REF, RUN / 'all-captured-center-side-inputs.jsonl', RUN / 'epsilon-grid-inputs.jsonl', RUN / 'epsilon-fd-comparison.jsonl')},
              'projection_passes': passes}
    (OUT / 'report.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False) + '\n')
    (OUT / 'manifest.json').write_text(json.dumps({str(q.relative_to(OUT)): sha(q) for q in OUT.iterdir() if q.is_file()}, indent=2) + '\n')
    print(json.dumps(clean({k: report[k] for k in ('paper_endpoint_count', 'strict_reference_passes', 'strict_reference_failures', 'epsilon_table')}), indent=2))

if __name__ == '__main__': main()
