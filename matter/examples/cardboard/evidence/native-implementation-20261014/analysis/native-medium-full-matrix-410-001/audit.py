from pathlib import Path
import hashlib
import json
import numpy as np

base = Path('/Users/home/cardboard-implementation-20261014')
out = base / 'analysis/native-medium-full-matrix-410-001'
out.mkdir(exist_ok=False)
reference_path = base / 'analysis/medium-tangent-state-sensitivity-1634-001/report.json'
reference = json.loads(reference_path.read_text())
case = base / 'native/two-stage-controlled-material-sweep-410'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def f32(words):
    return np.asarray([int(w, 16) for w in words], dtype='<u4').view('<f4').astype(float)
def comparison(actual, expected):
    delta = actual - expected
    norm = float(np.linalg.norm(expected))
    return dict(reference_l2=norm, absolute_l2=float(np.linalg.norm(delta)),
        max_absolute=float(np.max(abs(delta))),
        relative_l2=float(np.linalg.norm(delta) / norm) if norm else None,
        reference_zero_component_count=int(np.count_nonzero(expected == 0)),
        componentwise_absolute_difference=delta.tolist())
receipt = json.loads((case / 'receipt.json').read_text())
assert receipt['exit_code'] == 0 and receipt['source_inputs_unchanged'] and not receipt['timed_out']
bound = []
for path in (reference_path, base / 'analysis/medium-tangent-state-sensitivity-1634-001/manifest.json',
             case / 'prediction.json', case / 'receipt.json', case / 'inputs/external/source-build.json'):
    bound.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
policies = {}
for policy in ('default', 'accurate'):
    path = case / 'output' / f'paper-derivative-sweep-{policy}.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    header, footer = rows[0], rows[-1]
    assert header['rows'] == footer['rows'] == 1536
    assert header['capture_sha256'] == reference['bindings']['capture_sha256']
    bound.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
    points = []
    for ref in reference['results']:
        point = ref['point']
        env, tet = point['environment'], point['tet_id']
        selected = [r for r in rows[1:-1] if r['point']['environment'] == env
                    and r['point']['tet_id'] == tet]
        assert len(selected) == 192
        actions = {r['basis']: r for r in selected if r['kind'] == 'basis_action'}
        assert sorted(actions) == list(range(9))
        words = point['input_words']
        matrix_columns = []
        for basis in range(9):
            row = actions[basis]
            assert row['stress_valid'] and row['tangent_valid'] and row['projection_valid']
            raw = row['point']['input_f32_bits']
            assert raw[:9] == words['F_high_f32_bits']
            assert raw[9:18] == words['F_low_f32_bits']
            assert row['point']['accepted_state_f32_bits'] == words['accepted_state_f32_bits']
            direction = f32(raw[27:36])
            assert np.array_equal(direction, np.eye(9)[basis])
            rate_direction = f32(raw[36:45])
            expected_rate = np.zeros(9)
            expected_rate[basis] = np.float32(1.0 / f32(raw)[45])
            assert np.array_equal(rate_direction, expected_rate)
            matrix_columns.append(f32(row['output_f32_bits'])[26:35])
        matrix = np.stack(matrix_columns, axis=1)
        mp_matrix = np.asarray(ref['nine_direction_check']['mp90_tangent_matrix'])
        fp64_matrix = np.asarray(ref['nine_direction_check']['fp64_tangent_matrix'])
        cols = [comparison(matrix[:, i], mp_matrix[:, i]) for i in range(9)]
        full = comparison(matrix, mp_matrix)
        points.append(dict(environment=env, tet_id=tet,
            basis_order=header['basis_order'], input_and_accepted_history_words_identical=True,
            native_algorithmic_matrix=matrix.tolist(), independent_mp90_matrix=mp_matrix.tolist(),
            native_vs_mp90=full, native_vs_safeguarded_fp64=comparison(matrix, fp64_matrix),
            column_comparisons=cols,
            resolved_nonzero_column_norm_gate_1e_3=all(c['relative_l2'] is not None
                and c['relative_l2'] <= 1e-3 for c in cols),
            full_matrix_norm_gate_1e_3=full['relative_l2'] <= 1e-3,
            absolute_noise_floor=None))
    policies[policy] = dict(header=header, points=points)
report = dict(schema='numi.cardboard.native-medium-full-matrix-mp90.v1',
    source_run=str(case), source_bindings=bound, policies=policies,
    comparison='Nine production algorithmic actions at each of two exact captured medium centers versus independently projected MP90 matrices, widened to FP64 for comparison.',
    parameter_identity='The source-bound native checker verifies all 56 compiled FP32 parameter words against the exact captured header before dispatch; the independent reference binds the same capture. Both reference equations and this material are rate independent; native Hdot=H/dt is retained and checked.',
    limits=['Two medium centers only; no liner, all-point, element, assembled or physical trajectory qualification.',
        'Nonzero vector/matrix norm gates do not establish componentwise near-zero accuracy or an absolute noise floor.',
        'This compares native algorithmic derivatives to an independent analytic reference. Native stress finite-difference convergence remains a separate measurement.',
        'No branch-crossing or generic constitutive qualification is inferred.'],
    physical_steps=0, physical_validation=False)
(out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
(out / 'audit.py').write_bytes(Path(__file__).read_bytes())
(out / 'README.md').write_text('# Native full material matrices at two medium centers\n\n'
    'All nine production algorithmic basis actions from native410 are assembled with explicit '
    'row-major columns and compared against the independently projected MP90 and safeguarded '
    'FP64 matrices from the sealed reference package. Exact F high/low, accepted history, '
    'directions, rate directions and captured-source identity are checked before comparison. '
    'Nonzero column norm and full matrix norm results remain separate from unresolved '
    'near-zero component accuracy, native finite differences and complete nonlinear physics.\n')
(out / 'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.analysis-package.v1',
    artifacts={p.name:dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(out.iterdir()) if p.is_file()}, source_bindings=bound), indent=2) + '\n')
print(json.dumps({policy: {str(p['tet_id']): dict(relative=p['native_vs_mp90']['relative_l2'],
    worst_column_relative=max(c['relative_l2'] for c in p['column_comparisons']),
    norm_gate=p['resolved_nonzero_column_norm_gate_1e_3']) for p in value['points']}
    for policy, value in policies.items()}, indent=2))
