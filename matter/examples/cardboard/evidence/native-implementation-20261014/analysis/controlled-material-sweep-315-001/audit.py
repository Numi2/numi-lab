from pathlib import Path
import hashlib
import json
import numpy as np

base = Path('/Users/home/cardboard-implementation-20261014')
case = base / 'native/finer-controlled-material-sweep-315'
out = base / 'analysis/controlled-material-sweep-315-001'
out.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def floats(words):
    return np.array([int(w, 16) for w in words], dtype='<u4').view('<f4').astype(float)
def metric(actual, expected):
    norm = float(np.linalg.norm(expected))
    difference = actual - expected
    return dict(reference_norm=norm, absolute_l2=float(np.linalg.norm(difference)),
                max_absolute=float(np.max(abs(difference))),
                relative_l2=float(np.linalg.norm(difference) / norm) if norm > 0 else None)
def valid(row):
    return row['stress_valid'] and row['tangent_valid'] and row['projection_valid']
policies = {}
for policy in ('default', 'accurate'):
    path = case / 'output' / f'paper-derivative-sweep-{policy}.jsonl'
    all_rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = all_rows[1:-1]
    assert all_rows[0]['rows'] == all_rows[-1]['rows'] == len(rows) == 1536
    points = {}
    for tet in sorted({r['point']['tet_id'] for r in rows}):
        selected = [r for r in rows if r['point']['tet_id'] == tet]
        assert len(selected) == 192
        centers = [r for r in selected if r['kind'] == 'center_repeat']
        assert len(centers) == 3
        center = centers[0]
        center_input = floats(center['point']['input_f32_bits'])
        center_output = floats(center['output_f32_bits'])
        repeats_equal = all(r['output_f32_bits'] == center['output_f32_bits'] for r in centers)
        assert repeats_equal
        for row in selected:
            assert row['point']['accepted_state_f32_bits'] == center['point']['accepted_state_f32_bits']
        actions = {r['basis']: r for r in selected if r['kind'] == 'basis_action'}
        assert sorted(actions) == list(range(9))
        matrix = np.stack([floats(actions[i]['output_f32_bits'])[26:35] for i in range(9)], axis=1)
        cases = []
        for epsilon in sorted({r['epsilon_s'] for r in selected if r['kind'] == 'basis_perturbation'}, reverse=True):
            columns, positive, negative = [], [], []
            entries = []
            all_valid = valid(center) and all(valid(r) for r in actions.values())
            for basis in range(9):
                sides = {r['sign']: r for r in selected if r['kind'] == 'basis_perturbation'
                         and r['basis'] == basis and r['epsilon_s'] == epsilon}
                assert sorted(sides) == [-1, 1]
                lo, hi = [floats(sides[sign]['output_f32_bits']) for sign in (-1, 1)]
                a, b = [floats(sides[sign]['point']['input_f32_bits']) for sign in (-1, 1)]
                for row in sides.values():
                    all_valid = all_valid and valid(row)
                h_real = ((b[:9] - a[:9]) + (b[9:18] - a[9:18])) / (2 * epsilon)
                hd_real = (b[18:27] - a[18:27]) / (2 * epsilon)
                h = np.eye(9)[basis]
                hd = np.zeros(9)
                hd[basis] = np.float32(1 / center_input[45])
                columns.append((hi[8:17] - lo[8:17]) / (2 * epsilon))
                positive.append((hi[8:17] - center_output[8:17]) / epsilon)
                negative.append((center_output[8:17] - lo[8:17]) / epsilon)
                entries.append(dict(basis=basis,
                    valid_minus=valid(sides[-1]), valid_plus=valid(sides[1]),
                    failure_minus=sides[-1]['projection_failure'], failure_plus=sides[1]['projection_failure'],
                    realized_direction_error=metric(h_real, h),
                    realized_rate_direction_error=metric(hd_real, hd),
                    centered_comparison=metric(columns[-1], matrix[:, basis]),
                    positive_one_sided_comparison=metric(positive[-1], matrix[:, basis]),
                    negative_one_sided_comparison=metric(negative[-1], matrix[:, basis])))
            central = np.stack(columns, axis=1)
            cases.append(dict(epsilon_s=epsilon, all_entries_valid=all_valid,
                centered_matrix_comparison=metric(central, matrix),
                positive_one_sided_matrix_comparison=metric(np.stack(positive, axis=1), matrix),
                negative_one_sided_matrix_comparison=metric(np.stack(negative, axis=1), matrix),
                entries=entries))
        points[str(tet)] = dict(samples=len(selected), center_repeats_bit_identical=repeats_equal,
            accepted_history_unchanged=True, invalid_samples=sum(not valid(r) for r in selected),
            center_valid=valid(center), cases=cases)
    policies[policy] = dict(input_sha256=sha(path), header=all_rows[0],
        invalid_samples=sum(not valid(r) for r in rows), points=points)
report = dict(schema='numi.cardboard.controlled-sweep-comparison.v1', run=str(case),
    receipt=json.loads((case / 'receipt.json').read_text()), policies=policies,
    interpretation='Native stress finite differences versus native algorithmic actions, with realized input directions. This is not an independent material reference.',
    branch_classification='Pending independent endpoint projection/yield/complementarity evaluation; centered differences are not admitted as smooth-case gates.',
    absolute_noise_floor=None,
    limitations=['Same-batch center repeatability is not an arithmetic noise floor.',
                 'Invalid projections remain failures, never excluded to declare qualification.',
                 'One-sided comparisons are recorded but branch-transition identification remains required.',
                 'No element, assembled, complete-cycle or mesh/time convergence claim.'])
(out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
(out / 'audit.py').write_bytes(Path(__file__).read_bytes())
(out / 'README.md').write_text(
    '# Controlled native material sweep315\n\n'
    'Both policies completed1536 evaluations, retaining exact input, accepted-state '
    'and65 output words per sample. Three center repeats are checked bitwise. '
    'Nine native basis actions are compared with centered and one-sided stress '
    'differences at ten perturbation sizes, with actual rounded direction errors. '
    'This compares production paths to each other. Independent endpoint branch '
    'classification and absolute arithmetic noise floors remain open; no smooth '
    'derivative gate or physical qualification is inferred.\n')
(out / 'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.analysis-package.v1', artifacts={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()}), indent=2) + '\n')
print(json.dumps({p: dict(invalid=v['invalid_samples'], points={tet: dict(
    invalid=r['invalid_samples'], comparisons=[dict(epsilon=c['epsilon_s'],
    relative=c['centered_matrix_comparison']['relative_l2'], valid=c['all_entries_valid'])
    for c in r['cases']]) for tet, r in v['points'].items()}) for p, v in policies.items()}, indent=2))
