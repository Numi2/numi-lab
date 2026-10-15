#!/usr/bin/env python3
"""Strict source101 capture-360/control-352 physical-output parity audit."""
from __future__ import annotations
import hashlib, json, re, stat
from pathlib import Path

BASE=Path('/Users/home/cardboard-implementation-20261014')
CASES={
 'diagnostic':'solver/reset-augmentation-first-failure-capture-360',
 'control':'solver/reset-order-augmentation-complete-coupon-352',
}
ALLOWED_ENV={
 'NM_FEM_NEWTON_TRACE_ROOT',
 'NM_FEM_OPERATOR_FD_SELECTOR',
 'NM_FEM_OPERATOR_FD_CAPTURE_PATH',
 'NM_FEM_REGIONAL_RAP_CAPTURE_ONLY',
 'NM_FEM_TRUE_RESIDUAL_ROOT',
 'NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR',
 'NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT',
}

def sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def readj(p): return json.loads(p.read_text())
def check_regular(p):
    st=p.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise ValueError(f'not a regular file: {p}')
def hash_tree(d):
    out={}
    for p in sorted(d.rglob('*')):
        if p.is_symlink(): raise ValueError(f'symlink in output tree: {p}')
        if p.is_dir(): continue
        check_regular(p)
        out[p.relative_to(d).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}
    return out

def load(label, rel):
    case=BASE/rel
    prediction=readj(case/'prediction.json')
    receipt=readj(case/'receipt.json')
    assert prediction['schema']=='numi.cardboard.native-prediction.v1'
    assert receipt['schema']=='numi.cardboard.native-run-receipt.v2'
    assert prediction['executed_build_verified'] is True
    assert receipt['execution_started'] is True and receipt['source_inputs_unchanged'] is True
    assert receipt['changed_inputs']==[] and receipt['timed_out'] is False
    assert receipt['termination_errors']==[]
    # Verify every archived source/runtime input against its sealed digest.
    archived={}
    for key, expected in prediction['inputs_sha256'].items():
        path=case/'inputs'/key
        check_regular(path)
        actual=sha(path.read_bytes())
        if actual!=expected: raise ValueError(f'{label}: input hash mismatch {key}')
        archived[key]=actual
    argv=prediction['argv']
    assert argv.count('--output')==1
    oi=argv.index('--output')
    output=case/'output'
    assert Path(argv[oi+1]).resolve()==output.resolve(strict=True)
    argv_noout=argv[:oi]+argv[oi+2:]
    files=hash_tree(output)
    result=readj(output/'result.json')
    assert result['schema']=='numi.cardboard.probe-result.v1'
    armijo=[]
    operator_fd=[]
    for line in (case/'stderr.log').read_text(errors='strict').splitlines():
        marker='fem_residual_armijo_iteration='
        if marker in line:
            item=json.loads(line[line.index(marker)+len(marker):])
            armijo.append(item)
        marker='fem_operator_fd_iteration='
        if marker in line:
            operator_fd.append(json.loads(line[line.index(marker)+len(marker):]))
    env=prediction['environment']
    if len(armijo)!=64: raise ValueError(f'{label}: expected 64 Armijo records, got {len(armijo)}')
    for n,item in enumerate(armijo):
        if item.get('root')!=int(env['NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT']):
            raise ValueError(f'{label}: Armijo root mismatch at record {n}')
        if item.get('newton_iteration')!=n: raise ValueError(f'{label}: iteration order mismatch')
    return {'case':case,'prediction':prediction,'receipt':receipt,'argv_noout':argv_noout,
            'files':files,'result':result,'armijo':armijo,'operator_fd':operator_fd,'archived':archived}

def main():
    a=load('diagnostic',CASES['diagnostic']); b=load('control',CASES['control'])
    pa,pb=a['prediction'],b['prediction']
    for field in ('source_snapshot','runtime_dir','build_provenance','external_inputs',
                  'inputs_sha256','executed_build_verified','source_binding_limitation'):
        if pa.get(field)!=pb.get(field): raise ValueError(f'source/build identity mismatch: {field}')
    if a['archived']!=b['archived']: raise ValueError('input archive hashes differ')
    if a['argv_noout']!=b['argv_noout']: raise ValueError('argv differs beyond output path')
    changed=sorted(k for k in set(pa['environment'])|set(pb['environment'])
                   if pa['environment'].get(k)!=pb['environment'].get(k))
    if set(changed)!=ALLOWED_ENV: raise ValueError(f'unexpected diagnostic environment diff: {changed}')
    # Validate all common solver controls that must not change.
    for key in set(pa['environment'])-ALLOWED_ENV:
        if pa['environment'][key]!=pb['environment'].get(key):
            raise ValueError(f'non-diagnostic environment differs: {key}')
    assert pa['environment']['NM_FEM_RESIDUAL_ARMIJO_ROOT']==pb['environment']['NM_FEM_RESIDUAL_ARMIJO_ROOT']=='all'
    assert pa['environment']['NM_FEM_TRUE_RESIDUAL_RESTART']==pb['environment']['NM_FEM_TRUE_RESIDUAL_RESTART']=='0'
    expected={
      'NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT':('2','0'),
      'NM_FEM_NEWTON_TRACE_ROOT':('2','0'),
      'NM_FEM_TRUE_RESIDUAL_ROOT':('2','0'),
      'NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR':('23','27'),
      'NM_FEM_OPERATOR_FD_SELECTOR':('2:23',None),
      'NM_FEM_REGIONAL_RAP_CAPTURE_ONLY':('0',None),
    }
    for key,(v1,v2) in expected.items():
        if (pa['environment'].get(key),pb['environment'].get(key))!=(v1,v2):
            raise ValueError(f'unexpected selector values {key}')
    expected_capture=(a['case']/'operator-capture').resolve()
    if Path(pa['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH']).resolve()!=expected_capture:
        raise ValueError('unexpected operator-capture path')
    if pb['environment'].get('NM_FEM_OPERATOR_FD_CAPTURE_PATH') is not None:
        raise ValueError('control unexpectedly has FD capture path')
    af,bf=a['files'],b['files']; names=sorted(set(af)|set(bf))
    file_diffs=[n for n in names if af.get(n)!=bf.get(n) or
      ((a['case']/'output'/n).read_bytes() != (b['case']/'output'/n).read_bytes())]
    identical=not file_diffs
    # These two captures intentionally select different physical control roots.
    # Keep both raw records and report the non-comparability instead of claiming
    # telemetry parity across different coupon states.
    aa=a['armijo']; ba=b['armijo']
    ignored_root_equal=[]
    for x,y in zip(aa,ba):
        xx=dict(x); yy=dict(y); xx.pop('root',None); yy.pop('root',None)
        ignored_root_equal.append(xx==yy)
    trace={
      'diagnostic_root':aa[0]['root'],'control_root':ba[0]['root'],
      'records_each':{'diagnostic':len(aa),'control':len(ba)},
      'iteration_ranges':{'diagnostic':[aa[0]['newton_iteration'],aa[-1]['newton_iteration']],
                          'control':[ba[0]['newton_iteration'],ba[-1]['newton_iteration']]},
      'all_armijo_records_equal_after_dropping_root_only':all(ignored_root_equal),
      'equal_record_count_after_dropping_root_only':sum(ignored_root_equal),
      'telemetry_comparable':False,
      'reason':'The selected trace roots differ (control step 2 vs control step 0), so rows describe different physical Newton states. They are retained separately and are not used as an on/off telemetry parity claim.',
      'diagnostic_root_first_env0':{k:aa[0]['environments'][0].get(k) for k in ('base_l2','trial_l2','accepted','solver_status_code')},
      'diagnostic_root_last_env0':{k:aa[-1]['environments'][0].get(k) for k in ('base_l2','trial_l2','accepted','solver_status_code')},
      'control_root_first_env0':{k:ba[0]['environments'][0].get(k) for k in ('base_l2','trial_l2','accepted','solver_status_code')},
      'control_root_last_env0':{k:ba[-1]['environments'][0].get(k) for k in ('base_l2','trial_l2','accepted','solver_status_code')},
    }
    capture_lines=[x for x in (a['case']/'stderr.log').read_text().splitlines()
                   if 'fem_regional_rap_capture_skipped=' in x or 'fem_operator_fd_skipped=' in x or 'fem_newton_trace_skipped=' in x]
    fd=a['operator_fd']
    fd_summary=[]
    for record in fd:
        fd_summary.append({
          'root':record.get('root'),'newton_iteration':record.get('newton_iteration'),
          'contract':record.get('contract'),
          'environments':[{
             'environment':index,
             'operator_status_code_by_epsilon':[pair.get('operator_status_code') for pair in e.get('epsilon_pairs',[])],
             'direction_nonzero_by_epsilon':[pair.get('direction_nonzero') for pair in e.get('epsilon_pairs',[])],
             'epsilon_pairs':[{
                'epsilon':pair.get('epsilon'),
                'position_delta_m':pair.get('position_delta'),
                'free_node_count':pair.get('free_node_count'),
                'operator_status_code':pair.get('operator_status_code'),
                'direction_nonzero':pair.get('direction_nonzero'),
                'input_integrity_flag':pair.get('input_integrity_flag'),
                'maximum_delta_velocity_m_s':pair.get('maximum_delta_velocity'),
                'maximum_position_perturbation_m':pair.get('maximum_position_perturbation'),
                'stage_status_codes_plus_minus':pair.get('stage_status_codes_plus_minus'),
                'minimum_j_plus_minus':pair.get('minimum_j_plus_minus'),
                'residual_plus_l2':pair.get('residual_plus_l2'),
                'residual_minus_l2':pair.get('residual_minus_l2'),
                'operator_action_l2':pair.get('operator_action_l2'),
                'centered_fd_l2':pair.get('centered_fd_l2'),
                'relative_error':pair.get('relative_error'),
                'admissible_for_derivative_comparison':(
                    pair.get('stage_status_codes_plus_minus') == [0, 0] and
                    pair.get('residual_plus_l2') is not None and
                    pair.get('residual_minus_l2') is not None and
                    pair.get('direction_nonzero') is True and
                    pair.get('input_integrity_flag') == 1
                ),
             } for pair in e.get('epsilon_pairs',[])],
          } for index,e in enumerate(record.get('environments',[]))],
        })
    raw_evidence={}
    for label,loaded in (('diagnostic',a),('control',b)):
        raw_evidence[label]={}
        paths={
            'prediction':'prediction.json','receipt':'receipt.json',
            'stdout':'stdout.log','stderr':'stderr.log',
            'operator_fd_capture':'operator-capture',
            'observations':'output/observations.csv','result':'output/result.json',
        }
        for name,relative in paths.items():
            p=loaded['case']/relative
            raw_evidence[label][name]={
                'present':p.is_file(),
                'bytes':p.stat().st_size if p.is_file() else None,
                'sha256':sha(p.read_bytes()) if p.is_file() else None,
            }
    report={
      'schema':'numi.cardboard.source101_capture360_control352_audit.v1',
      'comparison':'physical_output_byte_identical' if identical else 'physical_output_differs',
      'qualification':'not_qualified',
      'note':'Both are failed physical attempts. Byte parity does not make either run successful or physically qualified.',
      'attempts':{
       'diagnostic':{'path':str(a['case']),'exit_code':a['receipt']['exit_code'],'timed_out':a['receipt']['timed_out'],'wall_seconds':a['receipt']['wall_seconds'],'result_status':a['result']['status'],'accepted_steps':a['result']['accepted_steps'],'physical_validation':a['result']['physical_validation'],'failure':a['result']['failure']},
       'control':{'path':str(b['case']),'exit_code':b['receipt']['exit_code'],'timed_out':b['receipt']['timed_out'],'wall_seconds':b['receipt']['wall_seconds'],'result_status':b['result']['status'],'accepted_steps':b['result']['accepted_steps'],'physical_validation':b['result']['physical_validation'],'failure':b['result']['failure']},
      },
      'source_and_protocol':{
       'source_snapshot':pa['source_snapshot'],'runtime_dir':pa['runtime_dir'],'build_provenance':pa['build_provenance'],
       'prediction_input_hashes_equal':True,'archived_input_hashes_verified':True,
       'argv_equal_except_declared_output_path':True,'changed_environment_keys':changed,
       'all_other_environment_values_equal':True,
       'shared_armijo_algorithm_root':'all','true_residual_restart':'0',
       'source_build_provenance_sha256':a['archived'].get('external/source-build.json'),
      },
      'physical_output_files':{'count':len(af),'identical_file_count':len(set(af)&set(bf))-len(file_diffs),'byte_identical':identical,'differing_files':file_diffs,'hashes':{'diagnostic':af,'control':bf}},
      'armijo_trace':trace,
      'operator_fd_capture':{
        'record_count':len(fd_summary),
        'payload_path':pa['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH'],
        'payload_exists':(a['case']/'operator-capture').is_file(),
        'payload_bytes':(a['case']/'operator-capture').stat().st_size if (a['case']/'operator-capture').is_file() else None,
        'payload_sha256':sha((a['case']/'operator-capture').read_bytes()) if (a['case']/'operator-capture').is_file() else None,
        'records':fd_summary,
        'interpretation':'No residual action comparison is admitted for a record with plus/minus stage status 5 or unavailable residual norms; zero direction is inconclusive.',
      },
      'raw_evidence_files':raw_evidence,
      'capture_status_lines':capture_lines,
    }
    dest=BASE/'analysis/source101-capture360-352-parity-001/report.json'
    dest.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'comparison':report['comparison'],'output_file_count':len(af),'differing_files':file_diffs,'armijo_root_pair':[aa[0]['root'],ba[0]['root']],'records_equal_after_root_drop':trace['equal_record_count_after_dropping_root_only'],'report':str(dest)},indent=2))

if __name__=='__main__': main()
