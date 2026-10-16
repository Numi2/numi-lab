from pathlib import Path
import csv, json, hashlib, collections, datetime

Q = Path('/Users/n/cardboard-implementation-20261014-host-macmini')
P = Q / 'analysis/native535-source201-full-coupon-command-001'
C = Q / 'solver/fresh-source201-damped-cgls-complete-coupon-535'
S = Q / 'snapshots/source-regional-rap-tail-preserve-201'
B = Q / 'build-source201-coupon535-001'
D = Q / 'analysis/native535-first-failure-adjudication-001'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
readj = lambda p: json.loads(Path(p).read_text())
spec = readj(P/'command-spec.json')
frozen = readj(S/'frozen-source.json')
build = readj(B/'source-build.json')
assert spec['source_frozen_sha256'] == sha(S/'frozen-source.json')
assert spec['source_build_sha256'] == sha(B/'source-build.json')
assert len(frozen['source_sha256']) == 817
assert build['success'] and build['source_sha256'] == frozen['source_sha256']
source_diffs = [n for n,h in frozen['source_sha256'].items() if sha(S/n) != h]
artifact_diffs = [n for n,h in build['artifacts_sha256'].items() if sha(n) != h]
assert not source_diffs and not artifact_diffs
for m in spec['material_bindings']:
    assert sha(m['path']) == m['sha256']
receipt = readj(P/'execution-535/receipt.json')
case_receipt = readj(C/'receipt.json')
assert receipt['exit_code'] == 2 and not receipt['timed_out'] and not receipt['space_abort']
assert not receipt['errors'] and receipt['source_and_artifacts_unchanged']
assert case_receipt['exit_code'] == 2 and not case_receipt['timed_out']
result = readj(C/'output/result.json')
assert result['status'] == 'failed' and result['accepted_steps'] == 17
assert result['physical_validation'] is False
env = readj(C/'native-environment.json')
effective = env['contract']['effective']
for k,v in spec['environment'].items():
    assert effective[k] == {'set': True, 'value': v}, (k,effective.get(k),v)
assert spec['fresh_from_zero'] is True and spec['resume'] is False
assert spec['protocol']['controls'] == 60 and spec['stop_after_control'] is None
rows = list(csv.DictReader((C/'output/observations.csv').open(newline='')))
assert len(rows) == 36
row0 = next(r for r in rows if r['step']=='17' and r['environment']=='0' and r['arm']=='bent')
row1 = next(r for r in rows if r['step']=='17' and r['environment']=='1' and r['arm']=='held_reference')
assert row0['status_code']=='10' and row0['object_index']=='0' and row0['failing_index']=='34'
assert row0['step_accepted']=='0' and row1['status_code']=='0' and row1['step_accepted']=='1'
diag = [float(row0[f'status_diagnostic_{x}']) for x in 'xyzw']
base_resid, last_trial_resid, last_alpha, trial_count = diag
assert trial_count == 24.0
assert last_alpha == 2.0**-23
assert last_trial_resid > base_resid
stderr = C/'stderr.log'
cycles = collections.defaultdict(list)
count = 0
for line in stderr.open():
    if line.startswith('fem_true_residual_cycle='):
        d = json.loads(line.split('=',1)[1])
        cycles[(d['root'],d['newton_iteration'])].append(d)
        count += 1
assert count == 73728
assert len(cycles) == 18*64
def codes(root,n,env_idx=0):
    return [(x['environments'][env_idx]['solver_status_code'],
             x['environments'][env_idx]['operator_status_code'],
             x['environments'][env_idx]['true_l2'])
            for x in cycles[(root,n)]]
# The failing Newton iteration (34) still completes 64 true-residual restart cycles.
n34 = cycles[(17,34)]
assert len(n34) == 64 and n34[-1]['final_cycle'] is True
assert all(s==0 and o==0 for s,o,_ in codes(17,34))
e34 = n34[-1]['environments'][0]
assert abs(e34['true_l2']-1.19334089e-7) < 1e-15
assert abs(e34['rhs_l2']-1.63952757e-6) < 1e-14
# Status10 begins in the next Newton record, where residual callbacks are no longer valid.
n35 = cycles[(17,35)]
assert len(n35) == 64
assert all(s==10 and o==4 and r==-1 for s,o,r in codes(17,35))
assert all(s==10 and o==4 and r==-1
           for n in range(35,64) for s,o,r in codes(17,n))
# Earlier control steps completed without status failures in either environment.
for root in range(17):
    for n in range(64):
        assert all(s==0 and o==0 for s,o,_ in codes(root,n))
source_file = S/'matter/src/metal/fem.metalinc'
lines = source_file.read_text().splitlines()
def findline(fragment):
    for i,line in enumerate(lines,1):
        if fragment in line:
            return i
    raise AssertionError('source fragment absent: '+fragment)
source_map = {
 'base_norm_metric': findline('metrics[environment] = float4(norm, INFINITY, 0.0f, 0.0f);'),
 'trial_alpha_exp2': findline('const float proposedAlpha = exp2(-float(trialIndex));'),
 'trial_norm_metric': findline('state.y = valid && isfinite(norm2) ? sqrt(max(norm2, 0.0f)) : INFINITY;'),
 'armijo_requires_descent': findline('state.y < state.x &&'),
 'line_search_exhaustion': findline('} else if (trialIndex + 1u >= solver.nonlinearIterations.w) {'),
 'failure_diagnostic_record': findline('float4(state.x, state.y, alpha, float(trialIndex + 1u)));')
}
assert source_map == {'base_norm_metric':3717, 'trial_alpha_exp2':4419,
                      'trial_norm_metric':4618, 'armijo_requires_descent':4658,
                      'line_search_exhaustion':4702, 'failure_diagnostic_record':4734}
cmp_path = D/'compare-prefixes-vs521.json'
# This report was generated by the shipped source535 compare_prefixes.py.
cmp = readj(cmp_path)
assert cmp['common_observation_rows']==36 and cmp['identical_common_rows']==36
assert cmp['different_common_rows']==0
assert cmp['per_environment_accepted_bent_prefix_controls']['0']=={'source198_521':17,'source201_535':17}
assert cmp['per_environment_accepted_bent_prefix_controls']['1']=={'source198_521':0,'source201_535':0}
geo=cmp['common_accepted_geometry']
assert len(geo)==35 and all(x['identical'] for x in geo)
summary = {
 'schema':'numi.cardboard.native535-first-failure-adjudication.v1',
 'verdict':'FAILED_AT_FIRST_CONTROL_17_ARM IJO_EXHAUSTION'.replace(' ',''),
 'run':{
   'source_snapshot':str(S),'source_frozen_sha256':sha(S/'frozen-source.json'),
   'build_record_sha256':sha(B/'source-build.json'),'command_spec_sha256':sha(P/'command-spec.json'),
   'runner_package_manifest_sha256':sha(P/'package-manifest.json'),
   'execution_receipt_sha256':sha(P/'execution-535/receipt.json'),
   'case_receipt_sha256':sha(C/'receipt.json'),'source_count':817,
   'build_artifact_count':len(build['artifacts_sha256']),'source_inputs_unchanged':case_receipt['source_inputs_unchanged'],
   'exit_code':2,'timed_out':False,'space_abort':False,'wall_seconds':case_receipt['wall_seconds'],
   'result_status':result['status'],'accepted_steps':result['accepted_steps'],'physical_validation':False,
   'protocol_controls':60,'completed_rows':17,'fresh_from_zero':True,'resume':False
 },
 'first_actual_failure':{
   'control_step_zero_based':17,'environment':0,'arm':'bent',
   'microstep_solver_iteration':34,'csv_failing_index':34,'status_code':10,
   'failure_kind':'No primary residual-Armijo candidate accepted; nm_fem_armijo_decide records nonlinear-solver failure after exhaustion.',
   'diagnostic_layout_from_source':'[base free-DOF residual L2, last tested trial free-DOF residual L2, last trial alpha, number of trials]',
   'diagnostics':{'base_residual_l2':base_resid,'last_trial_residual_l2':last_trial_resid,
                  'last_trial_residual_minus_base':last_trial_resid-base_resid,
                  'relative_increase':(last_trial_resid-base_resid)/base_resid,
                  'last_trial_alpha':last_alpha,'trial_count':int(trial_count),
                  'last_alpha_is_2^-23':last_alpha==2.0**-23},
   'linear_solve_at_same_iteration':{
      'true_residual_l2_after_final_restart_replacement':e34['true_l2'],
      'rhs_l2':e34['rhs_l2'],'solver_status_code':e34['solver_status_code'],
      'operator_status_code':e34['operator_status_code'],'restart_cycles':len(n34),
      'interpretation':'Linear residual cycles were healthy; the failure occurred later in nonlinear line search for that same Newton correction.'
   },
   'source_evidence':{'file':str(source_file),'sha256':sha(source_file),'line_anchors':source_map},
   'observability_limit':'The run did not request per-trial trace capture. The final failure record confirms 24 trials and reports only the last trial residual/alpha; it does not retain the preceding 23 candidate norms.'
 },
 'status_latching':{
   'first_latched_newton_record':35,
   'post_failure_records':29,
   'observed_post_failure_solver_status_code':10,
   'observed_post_failure_operator_status_code':4,
   'post_failure_true_residual_value':-1,
   'interpretation':'These are status-gated invalid callbacks after the actual Newton-34 Armijo exhaustion, not later independent physical residual evaluations.'
 },
 'source198_521_comparison':{
   'report_path':str(cmp_path),'report_sha256':sha(cmp_path),
   'common_observation_rows':cmp['common_observation_rows'],
   'identical_common_rows':cmp['identical_common_rows'],
   'different_common_rows':cmp['different_common_rows'],
   'common_accepted_geometry_files':len(geo),'all_common_geometry_identical':all(x['identical'] for x in geo),
   'accepted_bent_prefix_controls':cmp['per_environment_accepted_bent_prefix_controls'],
   'interpretation':'The retained common prefix is byte-identical to native521; this is descriptive prefix parity, not completion.'
 },
 'raw_bindings':{
   str(p):sha(p) for p in [
    P/'execution-535/launch.json',P/'execution-535/receipt.json',P/'execution-535/wrapper.log',
    C/'receipt.json',C/'native-environment.json',C/'stdout.log',C/'stderr.log',
    C/'output/result.json',C/'output/observations.csv',C/'output/manifest.json',
    C/'output/rejected-conditioning-000018/manifest.json',
    C/'output/rejected-conditioning-000018/candidate_nodes.bin',
    C/'output/rejected-conditioning-000018/factor_valid.bin'
   ]
 },
 'limits':[
  'This full-protocol diagnostic run failed at control step 17 and is not a completed coupon.',
  'The accepted prefix does not qualify the full trajectory.',
  'No mesh/time convergence, calibration, physical assembly, shipping-scale or qualification-axis claim follows.',
  'The run exposes Armijo exhaustion, but not the complete 24-trial residual curve or a causal explanation for its non-descent.'
 ]
}
(D/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({'report':str(D/'report.json'),'report_sha256':sha(D/'report.json'),
                  'first_failure':summary['first_actual_failure'],'source198_comparison':summary['source198_521_comparison']},indent=2))
