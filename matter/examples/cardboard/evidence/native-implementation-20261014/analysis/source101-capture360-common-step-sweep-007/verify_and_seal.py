from pathlib import Path
import hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
reference = BASE / 'analysis/source101-capture360-v3-uniform-1728-006'
reports, domains, metrics, inputs = [], [], {}, {}
for target in (1e-6, 3e-7, 1e-7):
    root = reference / 'output-006' if target == 1e-7 else HERE / ('output-' + format(target, '.1e'))
    report = json.loads((root / 'report.json').read_text())
    rows = [json.loads(s) for s in (root / 'all1728_v3_rows.jsonl').read_text().splitlines()]
    assert len(rows) == 1728
    ids = {int(r['tet_id']) for r in rows}
    assert len(ids) == 1728
    domains.append(ids)
    errors, exceeded, unstable, unresolved = [], [], [], []
    for row in rows:
        tid = int(row['tet_id']); v = row['continuation_v3']
        cases = [c for c in v['epsilon_cases'] if c['target_deltaF_frobenius'] == target]
        assert len(cases) == 1
        c = cases[0]
        for d in (v['center_projection'], c['plus']['continuation'], c['minus']['continuation']):
            fm = d['final_metrics']
            if not (d['converged'] and fm['scaled_residual'] <= 2e-10 and
                    fm['dynamic_state_scaled_correction_inf'] <= 2e-8): unresolved.append(tid)
        for label in ('plus', 'minus'):
            assert abs(c[label]['actual_deltaF_frobenius'] / target - 1) < 1e-7
        action = np.asarray(c['central_fd_action_Pa_per_s'])
        tangent = np.asarray(v['center_tangent_action_Pa_per_s'])
        denominator = float(np.linalg.norm(tangent))
        assert denominator > 0
        absolute = float(np.linalg.norm(action - tangent)); relative = absolute / denominator
        assert np.isclose(relative, c['central_fd_vs_fp64_tangent']['relative_l2'], rtol=2e-9, atol=1e-15)
        assert np.isclose(absolute, c['central_fd_vs_fp64_tangent']['absolute_l2'], rtol=2e-9, atol=1e-12)
        if relative > 1e-6: exceeded.append(tid)
        if not c['branch_signature_stable']: unstable.append(tid)
        errors.append((relative, tid)); metrics.setdefault(tid, {})[format(target, '.1e')] = relative
    expected = report['summary']['uniform_v3_all1728']
    assert sorted(exceeded) == sorted(expected['exceed_tet_ids'])
    assert sorted(set(unresolved)) == sorted(expected['v3_projection_gate_failure_tet_ids'])
    assert sorted(unstable) == sorted(expected['branch_change_tet_ids'])
    reports.append(dict(target_deltaF_norm=target, rows=len(rows), within_1e6=len(rows)-len(exceeded),
                        exceeded=exceeded, unresolved=sorted(set(unresolved)), observed_branch_changes=unstable,
                        maximum_relative_error=max(errors)[0], maximum_error_tet=max(errors)[1],
                        warning_count=len(report['warnings']), elapsed_seconds=report['elapsed_seconds']))
    for name, binding in report['bindings'].items():
        p = Path(binding['path']); assert sha(p) == binding['sha256']; inputs[str(p)] = binding['sha256']
    if target == 1e-7:
        for name in ('report.json', 'all1728_v3_rows.jsonl'):
            p = root / name; inputs[str(p)] = sha(p)
assert domains[0] == domains[1] == domains[2]
comparison = dict(schema='numi.cardboard.common-step-fp64-diagnostic.v1', steps=reports,
                  all_projection_checks_pass=all(not r['unresolved'] for r in reports),
                  all_observed_branch_predicates_stable=all(not r['observed_branch_changes'] for r in reports),
                  no_common_declared_step_meets_every_row=all(r['exceeded'] for r in reports),
                  original006_gate_unchanged=True, native_noise_floor_measured=False,
                  physical_qualification=False)
(HERE / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
(HERE / 'row-comparison.jsonl').write_text(''.join(json.dumps(dict(tet_id=k, nominal_direction_relative_errors=v), sort_keys=True) + '\n' for k,v in sorted(metrics.items())))
files = {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
manifest = dict(schema='numi.cardboard.common-step-fp64-package.v1', files=files, immutable_external_inputs=inputs,
                source006_manifest_sha256=sha(reference / 'manifest.json'))
path = HERE / 'manifest.json'
encoded = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
if path.exists(): assert path.read_text() == encoded
else: path.write_text(encoded)
print(json.dumps(comparison, indent=2))
