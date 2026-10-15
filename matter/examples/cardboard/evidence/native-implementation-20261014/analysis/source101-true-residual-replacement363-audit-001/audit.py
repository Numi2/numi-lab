from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import shutil

B = Path('/Users/home/cardboard-implementation-20261014')
O = B / 'analysis/source101-true-residual-replacement363-audit-001'
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
helper = B / 'analysis/source101-capture360-352-parity-001/compare.py'
spec = importlib.util.spec_from_file_location('retained_parity', helper)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
a = m.load('360', 'solver/reset-augmentation-first-failure-capture-360')
c = m.load('363', 'solver/selected-failure-true-residual-replacement-363')
assert a['archived'] == c['archived']
assert a['argv_noout'] == c['argv_noout']
assert a['receipt']['exit_code'] == c['receipt']['exit_code'] == 2
assert a['result'] == c['result'] and c['result']['accepted_steps'] == 2
assert a['armijo'][:23] == c['armijo'][:23]
changed_env = {k: [a['prediction']['environment'].get(k),
                   c['prediction']['environment'].get(k)]
               for k in set(a['prediction']['environment']) | set(c['prediction']['environment'])
               if a['prediction']['environment'].get(k) != c['prediction']['environment'].get(k)}
assert set(changed_env) == {'NM_FEM_NEWTON_TRACE_ROOT', 'NM_FEM_OPERATOR_FD_SELECTOR',
    'NM_FEM_OPERATOR_FD_CAPTURE_PATH', 'NM_FEM_REGIONAL_RAP_CAPTURE_ONLY',
    'NM_FEM_TRUE_RESIDUAL_RESTART'}
assert changed_env['NM_FEM_TRUE_RESIDUAL_RESTART'] == [None, '1']
assert a['prediction']['environment']['NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR'] == \
       c['prediction']['environment']['NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR'] == '23'
assert set(a['files']) == set(c['files'])
different = [n for n in a['files'] if a['files'][n] != c['files'][n]]
assert different == ['observations.csv']
with (a['case'] / 'output/observations.csv').open(newline='') as f:
    old = list(csv.DictReader(f))
with (c['case'] / 'output/observations.csv').open(newline='') as f:
    new = list(csv.DictReader(f))
assert len(old) == len(new) == 6
assert old[:4] == new[:4]
observed_changes = []
for row_index, (x, y) in enumerate(zip(old, new)):
    for key in x:
        if x[key] != y[key]:
            observed_changes.append(dict(row=row_index, key=key, baseline=x[key], experiment=y[key]))
assert all(d['row'] >= 4 and d['key'] in {'fgmres_iterations', 'status_diagnostic_y'}
           for d in observed_changes), observed_changes

def cycles(run):
    rows = [json.loads(line.split('=', 1)[1])
            for line in (run / 'stderr.log').read_text().splitlines()
            if line.startswith('fem_true_residual_cycle=')]
    assert len(rows) == 64
    assert [x['restart_cycle'] for x in rows] == list(range(64))
    assert all(x['root'] == 2 and x['newton_iteration'] == 23 for x in rows)
    return rows

original = cycles(a['case'])
replaced = cycles(c['case'])
assert all(x['restart_replacement'] is True for x in replaced)
assert all(x['restart_replacement'] is False for x in original)
assert all(e['operator_status_code'] == e['solver_status_code'] == 0
           for x in replaced for e in x['environments'])
target = 0.015625 * replaced[0]['environments'][0]['rhs_l2']
assert all(x['environments'][0]['diagnostics'][2] == 0 for x in replaced)
assert min(x['environments'][0]['true_l2'] for x in replaced) > target
rows = []
for x in replaced:
    e = x['environments'][0]
    rows.append(dict(restart_cycle=x['restart_cycle'], used_columns=e['diagnostics'][3],
        arnoldi_estimate_Ns=e['arnoldi_estimate'], fresh_true_residual_Ns=e['true_l2'],
        forcing_target_Ns=target, recomputed_convergence_flag=e['diagnostics'][2]))
with (O / 'residual-cycles.csv').open('x', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
e0 = c['armijo'][23]['environments'][0]
assert e0['solver_status_code'] == 10 and e0['solver_status']['failing_index'] == 23
assert e0['accepted'] is False and e0['diagnostics'][3] == 24
report = dict(schema='numi.cardboard.true-residual-replacement363-audit.v1',
    status='REPLACEMENT_EXECUTED_NONLINEAR_FAILURE_REMAINS',
    original_run=str(a['case']), experiment_run=str(c['case']),
    source_snapshot=a['prediction']['source_snapshot'],
    same_archived_inputs=True, same_physical_argv_except_output=True,
    changed_environment=changed_env, accepted_steps=2, required_steps=27,
    exact_preselected_newton_armijo_rows=23,
    exact_ordinary_files_except_observations=36,
    observation_changes=observed_changes, all_native_operator_statuses_healthy=True,
    forcing_target_Ns=target,
    selected_newton_actual_columns=sum(x['used_columns'] for x in rows),
    original_selected_newton_actual_columns=sum(x['environments'][0]['diagnostics'][3]
        for x in original if x['restart_cycle'] <= 6),
    reported_status_fgmres_iterations=e0['solver_status']['fgmres_iterations'],
    status_counter_semantics='Maximum statically scheduled iterationOffset+used, not the sum of formed columns after partial restarted cycles.',
    minimum_fresh_residual_Ns=min(x['fresh_true_residual_Ns'] for x in rows),
    final_fresh_residual_Ns=rows[-1]['fresh_true_residual_Ns'],
    selected_newton_armijo=e0,
    limitations=['This experiment replaces using the fresh uncached path while Arnoldi uses the cached path.',
        'Native368 independently measures both paths; source105 subsequently certifies using the same configured operator.',
        'Original material FD perturbations are infeasible; neither run qualifies an independent tangent.',
        'No complete coupon, mesh/time, component calibration, shipping assembly or crush/impact qualification.'],
    inputs_sha256={str(p): sha(p) for p in [helper,
        a['case']/'prediction.json', a['case']/'receipt.json', a['case']/'stderr.log',
        c['case']/'prediction.json', c['case']/'receipt.json', c['case']/'stderr.log']})
(O / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
shutil.copy2(helper, O/'retained_parity_helper.py')
shutil.copy2(__file__, O/'audit.py')
files = [dict(path=str(p.relative_to(O)), bytes=p.stat().st_size, sha256=sha(p))
         for p in sorted(O.rglob('*')) if p.is_file()]
(O/'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.analysis-evidence.v1',
    files=files, physical_transactions=0), indent=2)+'\n')
print(json.dumps({k: report[k] for k in ['status', 'selected_newton_actual_columns',
    'minimum_fresh_residual_Ns', 'final_fresh_residual_Ns', 'forcing_target_Ns']}, indent=2))
