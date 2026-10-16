from pathlib import Path
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
import re

B = Path('/Users/home/cardboard-implementation-20261014')
C = B / 'solver/cgls-dt-half-cycle-468'
O = B / 'analysis/native468-complete-failed-attempt-002'
O.mkdir(exist_ok=False)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text())

prediction = read(C / 'prediction.json')
receipt = read(C / 'receipt.json')
result = read(C / 'output/result.json')
guard = read(B / 'analysis/native468-launch-guard-001/receipt.json')
assert receipt['execution_started'] and receipt['source_inputs_unchanged']
assert receipt['exit_code'] == 2 and not receipt['timed_out']
assert not receipt['changed_inputs'] and not receipt['termination_errors']
assert not guard['space_abort'] and guard['wrapper_exit_code'] == 2
assert guard['native_child_process_groups_signalled'] == []
assert result['status'] == 'failed' and result['accepted_steps'] == 20
assert result['physical_validation'] is False
for name, digest in prediction['inputs_sha256'].items():
    assert sha(C / 'inputs' / name) == digest, name

with (C / 'output/observations.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 42
arms = {}
for environment, arm in [(0, 'bent'), (1, 'held_reference')]:
    arm_rows = [r for r in rows if int(r['environment']) == environment]
    assert len(arm_rows) == 21
    assert [int(r['step']) for r in arm_rows] == list(range(21))
    assert all(r['arm'] == arm for r in arm_rows)
    assert all(r['phase'] == 'loading' for r in arm_rows)
    accepted = [r for r in arm_rows if r['step_accepted'] == '1']
    rejected = [r for r in arm_rows if r['step_accepted'] == '0']
    assert len(accepted) == (20 if environment == 0 else 21)
    assert all(r['status_code'] == '0' for r in accepted)
    assert all(r['certificate_raw_accepted_flag'] == '1' for r in accepted)
    assert all(r['right_grip_constrained'] == '1' for r in arm_rows)
    def maximum(name):
        return max(float(r[name]) for r in accepted)
    arms[arm] = {
        'environment': environment, 'attempted_controls': len(arm_rows),
        'accepted_controls': len(accepted), 'rejected_controls': len(rejected),
        'first_control': 0, 'last_attempted_control': 20,
        'last_accepted_control': int(accepted[-1]['step']),
        'last_accepted_target_angle_deg': float(accepted[-1]['target_angle_deg']),
        'last_accepted_measured_angle_deg': float(accepted[-1]['measured_right_grip_angle_deg']),
        'last_accepted_time_s': float(accepted[-1]['time_s']),
        'accepted_prefix_min_J': min(float(r['min_J']) for r in accepted),
        'accepted_prefix_peak_free_force_imbalance_l2_N': maximum('free_force_imbalance_l2_N'),
        'accepted_prefix_peak_free_force_imbalance_max_N': maximum('free_force_imbalance_max_N'),
        'accepted_prefix_peak_certificate_residual': maximum('certificate_residual'),
        'accepted_prefix_peak_kinetic_energy_J': maximum('kinetic_energy_J'),
        'last_attempted_observation': arm_rows[-1],
    }
assert arms['bent']['last_attempted_observation']['status_code'] == '10'
assert arms['bent']['last_attempted_observation']['certificate_raw_accepted_flag'] == '0'
assert math.isclose(float(arms['bent']['last_attempted_observation']['target_angle_deg']), .328125)

counts = Counter()
timings = []
last_linear = {}
cgls = []
parse_errors = []
pattern = re.compile(r'^([a-zA-Z0-9_]+)=(\{.*\})\s*$')
with (C / 'stderr.log').open() as f:
    for number, line in enumerate(f, 1):
        match = pattern.match(line)
        if not match:
            continue
        label = match[1]
        try:
            event = json.loads(match[2])
        except json.JSONDecodeError as error:
            parse_errors.append({'line': number, 'label': label, 'error': str(error)})
            continue
        counts[label] += 1
        if label == 'cardboard_step_timing' and event.get('stage') == 'completed':
            timings.append(event)
        if label == 'fem_true_residual_cycle' and event.get('root') == 20:
            last_linear[event['newton_iteration']] = event
        if label == 'fem_base_cgls_direction' and event.get('root') == 20:
            cgls.append(event)
assert not parse_errors
assert [r['control_step'] for r in timings] == list(range(21))
assert all(r['command_buffer_status'] == 4 for r in timings)
assert sorted(last_linear) == list(range(64))
assert [r['newton_iteration'] for r in cgls] == list(range(64))
primary_cgls = [r for r in cgls if r.get('selection_policy') == 'base_cgls_primary_policy']
assert [r['newton_iteration'] for r in primary_cgls] == list(range(48,64))
assert all(r['environments'][0]['primary_policy_eligible'] is True for r in primary_cgls)
linear_history = []
for newton, event in sorted(last_linear.items()):
    bent = event['environments'][0]
    linear_history.append({
        'newton_iteration': newton, 'restart_cycle': event['restart_cycle'],
        'base_rhs_l2_Ns': bent['rhs_l2'], 'linear_true_residual_l2_Ns': bent['true_l2'],
        'linear_true_max_abs_Ns': bent['true_max_abs'],
        'solver_status_code': bent['solver_status_code'],
        'operator_status_code': bent['operator_status_code'],
    })

conditioning_path = C / 'output/rejected-conditioning-000021/manifest.json'
conditioning = read(conditioning_path)
assert conditioning['accepted_physical_state'] is False
assert conditioning['control_step'] == 20
for row in conditioning['files']:
    path = conditioning_path.parent / row['name']
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
material = read(C / 'output/material_state_000021_bent.json')
assert material['status_code'] == 10
assert material['solver_certificate_raw_accepted_flag'] == 0
assert 'rolled-back' in material['note']

outputs = {}
for path in sorted(C.rglob('*')):
    if path.is_file() and 'inputs' not in path.relative_to(C).parts:
        outputs[str(path.relative_to(C))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
live_environment = B / 'analysis/native468-live-solver-environment-001/report.json'
report = {
    'schema': 'numi.cardboard.native468_complete_failed_attempt.v1',
    'classification': 'Complete terminal failed attempt; incomplete physical loading cycle',
    'run': str(C), 'receipt': receipt,
    'capacity_guard': guard,
    'compiled_world_fingerprint': result['compiled_world_fingerprint'],
    'bound_input_files_verified': len(prediction['inputs_sha256']),
    'command004_sha256': guard['command_sha256'],
    'source157_executable_sha256': sha(Path(prediction['argv'][0])),
    'protocol': {'dt_s': float(rows[0]['time_s']), 'planned_controls': 60,
                 'loading':24, 'hold':6, 'unload':12, 'relax':6,
                 'release':6, 'post_release_relaxation':6,
                 'bend_angle_deg':.375, 'newton_iterations':64,
                 'residual_tolerance':5e-7, 'fgmres_iterations':1024,
                 'fgmres_restart':16, 'material_iterations':32,
                 'cgls_iterations':64, 'cgls_right_scale':'block3',
                 'cgls_primary_after':48},
    'arms': arms,
    'failure': {'control_step_zero_based':20, 'target_angle_deg':.328125,
                'status_code':10, 'status_name':'NM_STATUS_NONLINEAR_SOLVER_FAILURE',
                'final_nonlinear_residual':float(rows[-2]['certificate_residual']),
                'final_correction_certificate':float(rows[-2]['certificate_correction']),
                'nonlinear_residual_over_tolerance':float(rows[-2]['certificate_residual'])/5e-7,
                'no_timeout':True, 'no_capacity_abort':True,
                'completed_hold_withdrawal_release_relaxation':False},
    'production_trace_event_counts':dict(sorted(counts.items())),
    'failed_control_newton_linear_history':linear_history,
    'failed_control_cgls_history':cgls,
    'timing':{'control_steps':timings,
              'gpu_total_s':sum(r['gpu_wall_s'] for r in timings),
              'failed_control_gpu_s':timings[-1]['gpu_wall_s']},
    'rejected_diagnostic_bindings': {'conditioning_manifest_sha256':sha(conditioning_path),
                                    'conditioning_arrays_verified':len(conditioning['files']),
                                    'material_history':'accepted rolled-back history; not current projected Newton history'},
    'effective_environment_provenance':{
        'initial_contract':'legacy runner sealed explicit overrides and inherited ambient environment; full initial effective environment was not sealed',
        'live_observation_sha256':sha(live_environment),
        'live_observation_limit':'observed relevant solver environment during execution; cannot retroactively prove initial inheritance',
        'positive_pair_qualification_admitted':False},
    'comparison_limits':[
        'This is not a pure time-convergence pair with native467: outer iteration budget and nonlinear residual tolerance differ, and both cycles are incomplete.',
        'Accepted loading-prefix force, work, determinant and energy observations are descriptive only.',
        'Rejected CSV geometry/history fields are rolled back while residual certificates describe the failed candidate; the row is not one accepted physical state.',
        'Rejection conditioning buffers describe the last Newton linearization; no current projected history, exact final correction or native perturbation states were captured.',
        'The actual control20/Newton63 failure is distinct from original031 control19/Newton3; it does not reproduce or close that discrepancy.',
        'No new solver correctness, numerical convergence, component calibration, analogue validation, physical assembly, or crush/impact qualification follows.'
    ],
    'qualification':{'solver_correctness':'unqualified_complete_nonlinear_solve',
                     'numerical_convergence':'failed_incomplete_cycle',
                     'component_calibration':'unchanged_open',
                     'analogue_validation':'unchanged_open',
                     'physical_assembly_in_simulation':'unqualified',
                     'crush_impact_prediction':'unexecuted'},
    'physical_qualification':False,
    'terminal_files':outputs,
    'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
(O / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
(O / 'analyze.py').write_bytes(Path(__file__).read_bytes())
(O / 'README.md').write_text(
    '# Native468 complete failed attempt\n\n'
    'Source157 command004 completed with exit2 after 20 accepted bent loading controls; '
    'control20 at0.328125 degrees was rejected with nonlinear status10. '
    'No timeout or capacity abort occurred. All60 planned controls were not completed. '
    'Held-reference environment accepted21 controls independently; this does not admit the failed bent environment.\n\n'
    'The report hashes every terminal non-input file and verifies all894 sealed input aliases and all23 rejection-conditioning arenas. '
    'It retains every reported Newton true-residual cycle and CGLS summary in the original stderr; '
    'the report provides the failed-control per-Newton final-cycle records. '
    'Accepted geometry is distinct from rejected diagnostics. No convergence comparison or physical qualification is claimed.\n\n'
    'Full initial effective solver-environment provenance remains unavailable in this legacy run. '
    'The separately bound during-run environment observation does not retroactively close it. '
    'A future study must use a sanitized, explicitly sealed effective environment.\n')
manifest = {'schema':'numi.cardboard.native468_attempt_analysis_manifest.v1',
            'inputs':{str(C / name):sha(C / name) for name in ['prediction.json','receipt.json','output/result.json','output/observations.csv','stderr.log']},
            'files':{str(p.relative_to(O)):sha(p) for p in sorted(O.iterdir()) if p.is_file()},
            'physical_qualification':False}
(O / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({'report_sha256':sha(O/'report.json'), 'bound_inputs':len(prediction['inputs_sha256']),
                  'terminal_files':len(outputs), 'failure_control':20, 'accepted_bent':20,
                  'first_newton':linear_history[0], 'last_newton':linear_history[-1]}))
