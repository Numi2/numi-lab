#!/usr/bin/env python3
"""Source-bound full-contact on/off payload parity and accepted-state rollback audit."""
from pathlib import Path
import argparse
import json
import math
import re
import cardboard_diagnostic_history_compare as common

SCHEMA='numi.cardboard.full-contact-history-comparison.v1'
ALLOWED_DIAGNOSTICS=frozenset({'NM_FEM_NEWTON_TRACE_ROOT','NM_FEM_CONTACT_TRACE_ROOT',
    'NM_FEM_CONTACT_TRACE_ITERATION','NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT',
    'NM_FEM_TRUE_RESIDUAL_ROOT','NM_FEM_TRUE_RESIDUAL_RESTART'})
PERSISTENT_VECTORS=frozenset({'particles','femNodes','femFields','vascularState','vascularClock',
    'femTopologyNodes','femTopologyTetrahedra','cohesiveFaces','punctureChannels','topologyStates',
    'rigidStates','contactHistories','humanSupportHistories','humanSupportConsequences',
    'deformableContactHistories','particleMaterialState','femMaterialState','environmentParameters'})


def state_rows(attempt):
    root=attempt['path']/'diagnostic-state'
    rows=[]
    for path in sorted(root.glob('transaction-*.json')):
        row=common._read_json(path,'state attempt')
        if row.get('schema')!='numi.cardboard.full_contact_state_attempt.v1':
            raise common.ComparisonError('unsupported state attempt schema')
        index=row.get('attempt_index')
        if type(index) is not int or index!=len(rows) or path.name!=f'transaction-{index:06d}.json':
            raise common.ComparisonError('state attempt indices are incomplete or noncontiguous')
        for side in ('before','after'):
            if side=='after' and row.get('after_present') is not True:continue
            filename=row.get(side+'_file')
            if filename!=f'transaction-{index:06d}-{side}.bin':
                raise common.ComparisonError('unexpected raw state filename')
            common._safe_regular_file(root/filename)
            state=row.get(side,{})
            hashes=state.get('accepted_state_vector_sha256',{})
            counts=state.get('vector_counts',{})
            if set(hashes)!=PERSISTENT_VECTORS or not PERSISTENT_VECTORS<=set(counts):
                raise common.ComparisonError('accepted-state vector hash/count scope is incomplete')
            if any(not re.fullmatch('[0-9a-f]{64}',str(value)) for value in hashes.values()):
                raise common.ComparisonError('invalid vector SHA256')
        if row.get('accepted') is True:
            after=row.get('after',{})
            statuses=after.get('statuses',[])
            certificates=after.get('solver_certificates',[])
            if not statuses or len(statuses)!=len(certificates) or any(s.get('code')!=0 for s in statuses):
                raise common.ComparisonError('accepted state has failed/missing production status')
            if any(len(c.get('validity',[]))!=4 or not isinstance(c['validity'][3],(int,float)) or c['validity'][3]<=.5 for c in certificates):
                raise common.ComparisonError('accepted state lacks production certificate')
        rows.append(row)
    if not rows:raise common.ComparisonError('no complete state attempts')
    return rows


def rollback_audit(rows):
    result=[]
    for row in rows:
        record=dict(attempt_index=row['attempt_index'],accepted=row.get('accepted') is True)
        if record['accepted']:
            record['rollback']='not_applicable_accepted_transaction'
        elif row.get('after_present') is not True:
            record['rollback']='inconclusive_missing_after_state'
        else:
            before,after=row['before'],row['after']
            different=sorted(k for k in PERSISTENT_VECTORS
                if before['accepted_state_vector_sha256'][k]!=after['accepted_state_vector_sha256'][k]
                or before['vector_counts'][k]!=after['vector_counts'][k])
            cursor_changes=[key for key in ('control_step','physics_substep') if before.get(key)!=after.get(key)]
            record.update(rollback='persistent_scope_identical' if not different and not cursor_changes else 'persistent_scope_changed',
                          different_vectors=different,changed_cursors=cursor_changes)
        result.append(record)
    return result


def compare(diagnostic,control):
    left=common._load_attempt(Path(diagnostic),state_fixture=True)
    right=common._load_attempt(Path(control),state_fixture=True)
    for key in ('source_snapshot','runtime_dir','build_provenance','external_inputs','inputs_sha256','executed_build_verified','source_binding_limitation'):
        if left['prediction'].get(key)!=right['prediction'].get(key):
            raise common.ComparisonError(f'sealed source/build identity differs: {key}')
    if left['build']!=right['build'] or left['argv_without_output']!=right['argv_without_output']:
        raise common.ComparisonError('physical argv or source build differs')
    changed=sorted(k for k in set(left['environment'])|set(right['environment'])
                   if left['environment'].get(k)!=right['environment'].get(k))
    if set(changed)-ALLOWED_DIAGNOSTICS:
        raise common.ComparisonError('physical environment differs beyond capture selectors')
    if not changed or any(k in right['environment'] for k in ALLOWED_DIAGNOSTICS):
        raise common.ComparisonError('expected diagnostic-enabled left and disabled right')
    if left['environment'].get('NM_FEM_CONTACT_TRACE_ROOT')!='0' or not re.fullmatch(r'\d+',left['environment'].get('NM_FEM_CONTACT_TRACE_ITERATION','')):
        raise common.ComparisonError('expected bounded root0 contact/iteration diagnostic')
    # Restart replacement changes the numerical algorithm. Only the read-only
    # complete-operator instrument belongs in diagnostic on/off admission.
    requested_true = 'NM_FEM_TRUE_RESIDUAL_ROOT' in left['environment']
    if 'NM_FEM_TRUE_RESIDUAL_RESTART' in left['environment'] and (
            not requested_true or left['environment']['NM_FEM_TRUE_RESIDUAL_RESTART'] != '0'):
        raise common.ComparisonError('true residual replacement is a physical numerical algorithm difference')
    true_rows = []
    failed_production_trace_rows = 0
    if requested_true:
        if left['environment']['NM_FEM_TRUE_RESIDUAL_ROOT'] != '0':
            raise common.ComparisonError('expected bounded root0 true residual instrument')
        trace_path = left['path']/'stderr.log'
        if not trace_path.is_file():
            raise common.ComparisonError('requested true residual instrument was not observed: missing log')
        common._safe_regular_file(trace_path)
        for line in trace_path.read_text().splitlines():
            if not line.startswith('fem_true_residual_cycle='): continue
            row = json.loads(line.split('=',1)[1])
            if row.get('schema') != 2 or row.get('root') != 0 or row.get('restart_replacement') is not False or row.get('operator_scope') != 'full_fem_mass_bulk_cohesive_and_kinematic_rigid_self_contact':
                raise common.ComparisonError('true residual instrument lacks read-only full operator metadata')
            if not row.get('environments') or any(e.get('operator_status_code') != 0 for e in row['environments']):
                raise common.ComparisonError('true residual instrument operator failed')
            for environment in row['environments']:
                if environment.get('solver_status_code') != 0:
                    failed_production_trace_rows += 1
                    continue
                for metric in ('true_l2', 'arnoldi_estimate', 'true_max_abs', 'rhs_l2'):
                    value = environment.get(metric)
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise common.ComparisonError('true residual instrument has invalid metric: ' + metric)
            if all(e.get('solver_status_code') == 0 for e in row['environments']):
                true_rows.append(row)
        if not true_rows:
            raise common.ComparisonError('true residual instrument production solver failed before any valid metric' if failed_production_trace_rows else 'requested true residual instrument was not observed')
    if left['receipt']['exit_code']!=right['receipt']['exit_code']:
        raise common.ComparisonError('native fixture exit codes differ')
    left_rows,right_rows=state_rows(left),state_rows(right)
    names=set(left['output_files'])|set(right['output_files'])
    differing=[]
    for name in sorted(names):
        if (left['output_files'].get(name)!=right['output_files'].get(name) or
            not common._same_bytes(left['path']/'diagnostic-state'/name,right['path']/'diagnostic-state'/name)):
            differing.append(name)
    return dict(schema=SCHEMA,production_outputs_byte_identical=not differing,output_file_count=len(names),differing_files=differing,
        files=dict(diagnostic_enabled=left['output_files'],control=right['output_files']),
        sealed_identity=dict(inputs_sha256_equal=True,source_build_sha256=common._sha256(left['path']/'inputs/external/source-build.json'),
                             changed_diagnostic_environment_keys=changed,physical_argv_equal_except_output_directory=True),
        attempts={label:dict(exit_code=a['receipt']['exit_code'],timed_out=False,transactions=len(rows),accepted_transactions=sum(r.get('accepted') is True for r in rows),
                            rollback=rollback_audit(rows)) for label,a,rows in (('diagnostic_enabled',left,left_rows),('control',right,right_rows))},
        rollback_scope=sorted(PERSISTENT_VECTORS),strict_solver_qualification=False,physical_validation=False,
        true_residual_instrument=dict(requested=requested_true,
            healthy_production_rows=len(true_rows),
            failed_production_environment_rows_excluded=failed_production_trace_rows,
            metric_note='Rows after production failure are retained in raw logs and excluded from linear residual comparisons.'),
        qualification_note='Byte parity and rollback preserve failed attempt verdicts; status/certificate/contact sample and linearization scratch are outside the persistent rollback hash scope.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',required=True);p.add_argument('--control',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    try:r=compare(a.diagnostic,a.control);common.write_report_exclusive(a.output,r)
    except (common.ComparisonError,OSError) as error:p.error(str(error))
    print(json.dumps(dict(production_outputs_byte_identical=r['production_outputs_byte_identical'],differing_files=r['differing_files'],attempts=r['attempts'])))
    return 0 if r['production_outputs_byte_identical'] else 1


if __name__=='__main__':raise SystemExit(main())
