#!/usr/bin/env python3
"""Independent read-only integrity and trace audit for native468 report 002."""
from __future__ import annotations
from collections import Counter
import csv, hashlib, json, re
from pathlib import Path

B = Path('/Users/home/cardboard-implementation-20261014')
C = B / 'solver/cgls-dt-half-cycle-468'
A = B / 'analysis/native468-complete-failed-attempt-002'
OUT = B / 'analysis/native468-independent-audit-001'

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def check(condition: bool, text: str) -> None:
    if not condition:
        raise SystemExit('FAIL: '+text)

report=json.loads((A/'report.json').read_text())
manifest=json.loads((A/'manifest.json').read_text())
prediction=json.loads((C/'prediction.json').read_text())
receipt=json.loads((C/'receipt.json').read_text())
result=json.loads((C/'output/result.json').read_text())
guard=json.loads((B/'analysis/native468-launch-guard-001/receipt.json').read_text())
checks={}

# Reconfirm execution identity and terminal failure classification.
check(receipt['execution_started'] is True and receipt['source_inputs_unchanged'] is True,'run did not start from unchanged inputs')
check(receipt['exit_code']==2 and receipt['timed_out'] is False and receipt['termination_errors']==[],'receipt exit/timeout/termination fields disagree')
check(guard['space_abort'] is False and guard['wrapper_exit_code']==2 and guard['native_child_process_groups_signalled']==[],'capacity guard does not corroborate normal failed exit')
check(result['status']=='failed' and result['accepted_steps']==20 and result['physical_validation'] is False,'native result classification mismatch')
check(report['physical_qualification'] is False and report['qualification']['physical_assembly_in_simulation']=='unqualified','report overstates physical qualification')
checks['receipt_and_guard']={'exit_code':receipt['exit_code'],'timed_out':receipt['timed_out'],'space_abort':guard['space_abort'],'wall_seconds':receipt['wall_seconds']}

# Rehash every sealed input alias, not just manifest count.
input_rows=prediction['inputs_sha256']
for name,digest in input_rows.items():
    p=C/'inputs'/name
    check(p.is_file() and not p.is_symlink() and sha(p)==digest,'input hash mismatch: '+name)
check(len(input_rows)==894,'expected 894 bound input aliases')
check(report['bound_input_files_verified']==894,'report input count mismatch')
checks['inputs']={'count':len(input_rows),'all_sha256_match':True,'prediction_sha256':sha(C/'prediction.json')}

# CSV count, control prefix and rejected-candidate / rolled-back-row distinction.
with (C/'output/observations.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
check(len(rows)==42,'observation CSV row count mismatch')
arms={}
for env,arm,expected_accepts in ((0,'bent',20),(1,'held_reference',21)):
    rs=[r for r in rows if int(r['environment'])==env]
    check(len(rs)==21 and [int(r['step']) for r in rs]==list(range(21)),'arm row sequence mismatch: '+arm)
    check(all(r['arm']==arm and r['phase']=='loading' for r in rs),'arm/phase mismatch: '+arm)
    accepted=[r for r in rs if r['step_accepted']=='1']
    rejected=[r for r in rs if r['step_accepted']=='0']
    check(len(accepted)==expected_accepts,'accepted prefix mismatch: '+arm)
    check(all(r['status_code']=='0' and r['certificate_raw_accepted_flag']=='1' for r in accepted),'accepted row has nonaccepted certificate: '+arm)
    check(all(r['right_grip_constrained']=='1' for r in rs),'grip constraint missing: '+arm)
    arms[arm]={'attempted':len(rs),'accepted':len(accepted),'rejected':len(rejected),'last_accepted_step':int(accepted[-1]['step'])}
failed=next(r for r in rows if r['environment']=='0' and r['step']=='20')
last_accepted_bent=next(r for r in rows if r['environment']=='0' and r['step']=='19')
held=next(r for r in rows if r['environment']=='1' and r['step']=='20')
check(failed['status_code']=='10' and failed['certificate_raw_accepted_flag']=='0' and failed['step_accepted']=='0','bent rejected row mismatch')
check(float(failed['certificate_residual'])==report['failure']['final_nonlinear_residual'],'failure residual report mismatch')
check(float(failed['certificate_correction'])==report['failure']['final_correction_certificate'],'failure correction report mismatch')
check(held['status_code']=='0' and held['step_accepted']=='1','held control20 should be independent accepted reference row')
rollback_geometry_fields=('min_J','max_J','max_node_displacement_m','max_free_displacement_m',
                          'max_free_speed_m_s','kinetic_energy_J','max_material_state_change',
                          'measured_right_grip_angle_deg','current_fixed_nodes')
check(all(failed[k]==last_accepted_bent[k] for k in rollback_geometry_fields),
      'rejected-row geometry/material observables do not match the last accepted bent state')
check(failed['certificate_residual']!=last_accepted_bent['certificate_residual'] and
      failed['certificate_correction']!=last_accepted_bent['certificate_correction'],
      'failed candidate certificate was unexpectedly replaced by the prior accepted certificate')
failed_material=json.loads((C/'output/material_state_000021_bent.json').read_text())
check(failed_material['solver_certificate_raw_accepted_flag']==0 and
      failed_material['status_code']==10 and 'rolled-back' in failed_material['note'],
      'failed material record does not preserve rejected status and rollback semantics')
check(report['comparison_limits'][2].startswith('Rejected CSV geometry/history fields are rolled back while residual certificates describe the failed candidate'),'report does not distinguish rejected candidate residual from rollback geometry')
checks['csv_and_state_roles']={'rows':42,'arms':arms,'bent_failed_control':20,'failed_residual':float(failed['certificate_residual']),'held_control20_status':held['status_code'],'rejected_geometry_matches_last_accepted_fields':list(rollback_geometry_fields),'failed_certificate_remains_from_rejected_candidate':True,'rejected_row_is_not_one_accepted_physical_state':True}

# Exact terminal file inventory and each output hash.
actual={}
for p in sorted(C.rglob('*')):
    if p.is_file() and 'inputs' not in p.relative_to(C).parts:
        rel=str(p.relative_to(C))
        actual[rel]={'bytes':p.stat().st_size,'sha256':sha(p)}
recorded=report['terminal_files']
check(len(actual)==77 and len(recorded)==77,'terminal-file inventory should contain 77 files')
check(actual==recorded,'terminal file set/byte/hash mismatch')
check(set(manifest['files'])=={'README.md','analyze.py','report.json'},'analysis manifest file inventory changed')
for name,digest in manifest['files'].items():
    check(sha(A/name)==digest,'analysis package hash mismatch: '+name)
checks['terminal_files']={'count':len(actual),'exact_set_and_sha256_match':True,'analysis_manifest_files':len(manifest['files'])}

# All rejection-conditioning arrays are independently rehashed.
cond_path=C/'output/rejected-conditioning-000021/manifest.json'
cond=json.loads(cond_path.read_text())
check(cond['accepted_physical_state'] is False and cond['control_step']==20,'conditioning sidecar state/control mismatch')
for row in cond['files']:
    p=cond_path.parent/row['name']
    check(p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'conditioning array mismatch: '+row['name'])
check(len(cond['files'])==23 and report['rejected_diagnostic_bindings']['conditioning_arrays_verified']==23,'expected 23 conditioned arrays')
check(report['rejected_diagnostic_bindings']['conditioning_manifest_sha256']==sha(cond_path),'conditioning manifest binding mismatch')
check(report['rejected_diagnostic_bindings']['material_history']=='accepted rolled-back history; not current projected Newton history','report mislabels material history')
checks['conditioning']={'arrays':len(cond['files']),'all_hashes_match':True,'accepted_physical_state':False,'meaning':'rejected Newton linearization inputs, not accepted projected candidate state'}

# Independently decode complete stderr event histories.
pattern=re.compile(r'^([a-zA-Z0-9_]+)=(\{.*\})\s*$')
counts=Counter(); linears={}; cgls=[]; timings=[]; parse_errors=[]
with (C/'stderr.log').open() as f:
    for lineno,line in enumerate(f,1):
        m=pattern.match(line)
        if not m: continue
        try: event=json.loads(m[2])
        except json.JSONDecodeError as e:
            parse_errors.append((lineno,m[1],str(e))); continue
        label=m[1]; counts[label]+=1
        if label=='cardboard_step_timing' and event.get('stage')=='completed': timings.append(event)
        if label=='fem_true_residual_cycle' and event.get('root')==20: linears[event['newton_iteration']]=event
        if label=='fem_base_cgls_direction' and event.get('root')==20: cgls.append(event)
check(not parse_errors,'stderr emitted malformed structured JSON')
check(len(timings)==21 and [r['control_step'] for r in timings]==list(range(21)),'completed timing/control sequence mismatch')
check(all(r['command_buffer_status']==4 for r in timings),'a control command buffer did not complete')
check(sorted(linears)==list(range(64)),'failed-control Newton history lacks one final record per iteration')
check(len(cgls)==64 and [r['newton_iteration'] for r in cgls]==list(range(64)),'failed-control CGLS summaries are not 64 ordered Newton records')
primary=[r for r in cgls if r.get('selection_policy')=='base_cgls_primary_policy']
check([r['newton_iteration'] for r in primary]==list(range(48,64)),'primary CGLS selection window is not Newton 48..63')
check(all(r['environments'][0]['primary_policy_eligible'] is True for r in primary),'eligible bent environment not marked for each selected primary row')
reported_cgls=report['failed_control_cgls_history']
check(cgls==reported_cgls,'reported CGLS event rows differ from source stderr')
last_linear=[]
for i in range(64):
    e=linears[i]; env=e['environments'][0]
    last_linear.append({'newton_iteration':i,'restart_cycle':e['restart_cycle'],'base_rhs_l2_Ns':env['rhs_l2'],'linear_true_residual_l2_Ns':env['true_l2'],'linear_true_max_abs_Ns':env['true_max_abs'],'solver_status_code':env['solver_status_code'],'operator_status_code':env['operator_status_code']})
check(last_linear==report['failed_control_newton_linear_history'],'reported Newton last-cycle history differs from source stderr')
check(counts==Counter(report['production_trace_event_counts']),'event-count summary mismatch')
check(counts['fem_true_residual_cycle']==86016 and counts['fem_base_cgls_direction']==1344,'total production trace counts mismatch')
check(all(r['restart_cycle']==63 for r in last_linear),'reported Newton rows are not final recorded restart-cycle summaries')
checks['solver_traces']={'completed_control_timing_rows':len(timings),'failed_control_newton_final_cycle_rows':len(last_linear),'failed_control_cgls_summary_rows':len(cgls),'base_cgls_primary_newton_range':[48,63],'all_recorded_structured_events_parse':True,'total_true_residual_cycle_events':counts['fem_true_residual_cycle'],'total_cgls_direction_events':counts['fem_base_cgls_direction']}

# No assertion promotes the incomplete failed cycle into a convergence or physical claim.
check(report['classification']=='Complete terminal failed attempt; incomplete physical loading cycle','classification mismatch')
check(report['failure']['completed_hold_withdrawal_release_relaxation'] is False,'report claims all phases completed')
check(report['qualification']['numerical_convergence']=='failed_incomplete_cycle','numerical classification mismatch')
check(report['effective_environment_provenance']['positive_pair_qualification_admitted'] is False,'environment provenance was overstated')
check(report['protocol']['planned_controls']==60 and report['protocol']['newton_iterations']==64 and report['protocol']['cgls_iterations']==64,'solver/protocol metadata mismatch')
checks['qualification_limits']={'physical_qualification':False,'full_cycle_completed':False,'initial_full_environment_provenance':False,'claim':'failed incomplete cycle; accepted prefix only descriptive'}

out={
 'schema':'numi.cardboard.native468_independent_audit.v1',
 'classification':'independent integrity and trace checks pass; native attempt remains failed/incomplete and unqualified',
 'inputs':{
  'source_analysis_report_sha256':sha(A/'report.json'),
  'source_analysis_manifest_sha256':sha(A/'manifest.json'),
  'source_analyzer_sha256':sha(A/'analyze.py'),
  'prediction_sha256':sha(C/'prediction.json'),
  'receipt_sha256':sha(C/'receipt.json'),
  'native_result_sha256':sha(C/'output/result.json'),
  'observations_sha256':sha(C/'output/observations.csv'),
  'stderr_sha256':sha(C/'stderr.log'),
  'capacity_guard_sha256':sha(B/'analysis/native468-launch-guard-001/receipt.json'),
  'conditioning_manifest_sha256':sha(cond_path),
  'bound_input_count':len(input_rows),
  'source_executable_sha256':sha(Path(prediction['argv'][0]))},
 'checks':checks,
 'source_analyzer_limitations':[
   'The source analyzer does not explicitly assert the expected 23-array conditioning count, but this independent audit checks and rehashes all 23 entries.',
   'The source report summarizes one final recorded residual cycle per Newton iteration; it does not represent every restart row as 64 independent linear solves.',
   'The accepted-prefix and held-reference records are descriptive only; no full loading/hold/withdrawal/release/relaxation cycle completed.'
 ],
 'physical_qualification':False,
 'native_solver_convergence_qualification':False,
}
(OUT/'report.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report':str(OUT/'report.json'),'checks':{k:True for k in checks},'input_count':len(input_rows),'terminal_file_count':len(actual),'conditioning_arrays':len(cond['files']),'newton_rows':len(last_linear),'cgls_rows':len(cgls)},indent=2))
