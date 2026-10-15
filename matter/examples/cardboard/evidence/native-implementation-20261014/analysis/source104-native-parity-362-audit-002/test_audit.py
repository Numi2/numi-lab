#!/usr/bin/env python3
"""Synthetic contract tests for the source104-vs-source103 parity comparator."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit',HERE/'audit.py')
audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def expect_error(fn, needle):
    try:
        fn()
    except (ValueError, AssertionError) as exc:
        assert needle in str(exc), (needle,str(exc))
    else:
        raise AssertionError(f'expected failure containing {needle!r}')


def status(code, failing, diag):
    return {'code':code,'environment':0,'object_index':0,'failing_index':failing,
            'completed_microsteps':0,'fgmres_iterations':0,'contact_count':0,'event_count':0,
            'diagnostics':diag}


def trace_pair():
    base_env={
      'base_l2':1.0,'trial_l2':None,'trial_alpha':1.1920929e-7,'accepted':False,
      'certified_noop':False,'minimum_J':0.9999,'line_search_raw':[0.0,0.9999,0.0,7],
      'solver_status_code':10,
      'solver_status':status(10,7,[1.0,None,1.0e-7,24.0]),
      'trial_status':status(6,283,[92.0,127.0,0.0,0.0])}
    new_env=copy.deepcopy(base_env)
    new_env['trial_status']['diagnostics'][3]=3.0
    new_env.update({
      'trial_outcome':'last_tested_trial_rejected',
      'last_trial_status_role':'last_tested_trial_status',
      'base_health':'healthy_at_armijo_entry',
      'base_residual_interpretation':'complete_assembled_free_dof_residual',
      'pretrial_status':status(0,4294967295,[None,0.0,0.0,0.0])})
    old={'schema':1,'root':1201,'newton_iteration':7,'semantics':audit.COMMON_SEMANTICS_103,
         'environments':[base_env]}
    new={'schema':3,'root':1201,'newton_iteration':7,'semantics':audit.COMMON_SEMANTICS_104,
         'environments':[new_env]}
    old_enabled={'schema':1,'root':1201,'maximum_newton_records':64}
    new_enabled={'schema':3,'root':1201,'maximum_newton_records':64,'base_status_capture':True}
    complete={'schema':1,'root':1201,'iteration':7,'bytes':100,'arrays':145,'manifest':'/old/path'}
    complete2={'schema':1,'root':1201,'iteration':7,'bytes':100,'arrays':145,'manifest':'/new/path'}
    return old,new,old_enabled,new_enabled,complete,complete2


def main():
    # Synthetic status marker accepted only at the reviewed failure-status field.
    out=[]
    audit.compare_json_exact_with_marker({'trial_status':status(6,283,[92,127,0,0])},
        {'trial_status':status(6,283,[92,127,0,3])},'trace',out)
    assert len(out)==1
    expect_error(lambda:audit.compare_json_exact_with_marker({'metric':1.0},{'metric':1.0000001},'trace'),
                 'value differs')
    expect_error(lambda:audit.compare_json_exact_with_marker({'solver_status':status(10,7,[1,2,3,0])},
        {'solver_status':status(10,7,[1,2,3,3])},'trace'), 'value differs')
    expect_error(lambda:audit.compare_json_exact_with_marker({'diagnostics':[1,2,3,0]},
        {'diagnostics':[1,2,3,3]},'trace'), 'value differs')

    # Snapshot source identity and all state payload bytes stay exact; only the
    # known serialized device-program fingerprint may differ across rebuilds.
    prefix=(0).to_bytes(4,'little')+(0x1122334455667788).to_bytes(8,'little')
    old_snapshot=prefix+(0xaabbccddeeff0011).to_bytes(8,'little')+b'payload'
    new_snapshot=prefix+(0x1234567890abcdef).to_bytes(8,'little')+b'payload'
    assert audit.compare_snapshot_payload_bytes(old_snapshot,new_snapshot,'synthetic.bin')['all_remaining_payload_bytes_exact']
    expect_error(lambda:audit.compare_snapshot_payload_bytes(old_snapshot,
        prefix+(0x1234567890abcdef).to_bytes(8,'little')+b'altered','synthetic.bin'),
        'payload differs outside')
    expect_error(lambda:audit.compare_snapshot_payload_bytes(old_snapshot,
        (0).to_bytes(4,'little')+(0x1122334455667789).to_bytes(8,'little')+
        (0x1234567890abcdef).to_bytes(8,'little')+b'payload','synthetic.bin'),
        'source fingerprint/message changed')

    old,new,oe,ne,complete,complete2=trace_pair()
    with tempfile.TemporaryDirectory() as td:
        b=Path(td)/'b'; c=Path(td)/'c'; b.mkdir(); c.mkdir()
        (b/'stderr.log').write_text(
          'fem_residual_armijo_trace_enabled='+json.dumps(oe)+'\n'+
          'fem_coupled_capture_complete='+json.dumps(complete)+'\n'+
          'fem_residual_armijo_iteration='+json.dumps(old)+'\n')
        (c/'stderr.log').write_text(
          'fem_residual_armijo_trace_enabled='+json.dumps(ne)+'\n'+
          'fem_coupled_capture_complete='+json.dumps(complete2)+'\n'+
          'fem_residual_armijo_iteration='+json.dumps(new)+'\n'+
          'ordinary diagnostic line\n')
        (b/'stderr.log').write_text(
          'fem_residual_armijo_trace_enabled='+json.dumps(oe)+'\n'+
          'fem_coupled_capture_complete='+json.dumps(complete)+'\n'+
          'fem_residual_armijo_iteration='+json.dumps(old)+'\n'+
          'ordinary diagnostic line\n')
        report=audit.compare_trace_logs(b,c)
        assert report['armijo_records']==1 and report['base_status_capture'] is True
        assert len(report['marker_changes'])==1
        # A changed physical residual stays an error, even if schema 3 is internally coherent.
        changed=copy.deepcopy(new); changed['environments'][0]['base_l2']=1.0001
        (c/'stderr.log').write_text(
          'fem_residual_armijo_trace_enabled='+json.dumps(ne)+'\n'+
          'fem_coupled_capture_complete='+json.dumps(complete2)+'\n'+
          'fem_residual_armijo_iteration='+json.dumps(changed)+'\n'+
          'ordinary diagnostic line\n')
        expect_error(lambda:audit.compare_trace_logs(b,c),'value differs')
        # Invalid health classification is rejected instead of ignored as a schema addition.
        changed=copy.deepcopy(new); changed['environments'][0]['base_health']='unavailable_status_snapshot_missing'
        (c/'stderr.log').write_text(
          'fem_residual_armijo_trace_enabled='+json.dumps(ne)+'\n'+
          'fem_coupled_capture_complete='+json.dumps(complete2)+'\n'+
          'fem_residual_armijo_iteration='+json.dumps(changed)+'\n'+
          'ordinary diagnostic line\n')
        expect_error(lambda:audit.compare_trace_logs(b,c),'base_health disagrees')
        changed['environments'][0]['base_health']='healthy_at_armijo_entry'
        (c/'stderr.log').write_text(
          'fem_residual_armijo_trace_enabled='+json.dumps(ne)+'\n'+
          'fem_coupled_capture_complete='+json.dumps(complete2)+'\n'+
          'fem_residual_armijo_iteration='+json.dumps(changed)+'\n'+
          'unexpected extra output\n')
        expect_error(lambda:audit.compare_trace_logs(b,c),'unstructured native log lines differ')
    print('synthetic contract tests passed: exact numeric equality, scoped VT marker, schema-3 health, serialized fingerprint-only delta, altered payload rejection')

if __name__=='__main__': main()
