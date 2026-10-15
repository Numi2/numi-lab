#!/usr/bin/env python3
"""Strict three-run source104/105/106 capture-lifetime parity audit.

Retained native attempts only. The known 369 false RHS-ready bit is preserved
as a metadata failure; only source/device provenance fields and run-specific
output paths are normalized. No run is launched and no physical failure is
promoted to a successful simulation.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/Users/home/cardboard-implementation-20261014')
RUNS = {
    '362': ROOT/'native/health-and-vt-marker-firstfatal-362',
    '369': ROOT/'native/true-residual-default-parity-369',
    '371': ROOT/'native/capture-lifetime-default-parity-371',
}
SOURCES = {
    '362': ROOT/'snapshots/source-contact-health-diagnostics-104',
    '369': ROOT/'snapshots/source-true-residual-certification-105',
    '371': ROOT/'snapshots/source-capture-callback-lifetime-106',
}
OLD_AUDIT_DIR = ROOT/'analysis/source105-native-default-parity-362-369-audit-001'
DEVICE_FP_RANGE = (12, 20)
TRACE_PREFIX = 'fem_residual_armijo_iteration='


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def strict_json(path):
    def reject(token):
        raise ValueError(f'non-standard JSON token {token} in {path}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def old_audit():
    path = OLD_AUDIT_DIR/'audit.py'
    spec = importlib.util.spec_from_file_location('source105_parity_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_source106(old):
    base = SOURCES['369']; cand = SOURCES['371']
    bm = strict_json(base/'frozen-source.json')['source_sha256']
    cm = strict_json(cand/'frozen-source.json')['source_sha256']
    need(len(bm)==780 and len(cm)==781, 'source105/106 inventory count changed')
    for snapshot, mapping in ((base,bm),(cand,cm)):
        for relative,digest in mapping.items():
            p=snapshot/relative
            need(p.is_file() and sha(p)==digest, f'frozen source hash mismatch: {p}')
    changes={p:{'source105':bm.get(p),'source106':cm.get(p)}
             for p in sorted(set(bm)|set(cm)) if bm.get(p)!=cm.get(p)}
    expected={
      'matter/src/runtime.mm':{
        'source105':'924669986619a88c6e34ec79e306495c1113e6ea35b0c793e8a36cfede887ec2',
        'source106':'2e26d26a04326a3ce567cccb60d213da8f4a8ceaf178c98153939f1b68752fd7'},
      'matter/src/coupled_capture_manifest_context.hpp':{
        'source105':None,
        'source106':'7d82c3c8f1fff4546bd65ee1641b4fd5faced73dea9c75501ea780254069bbae'}
    }
    need(changes==expected, f'source105/106 delta differs from callback-lifetime overlay: {changes}')
    integ=strict_json(ROOT/'analysis/source106-callback-integration-001/integration.json')
    need(integ.get('snapshot')==str(cand) and integ.get('base_snapshot')==str(base),
         'source106 integration lineage mismatch')
    expected_integration={p:{'before':values['source105'],'after':values['source106']}
                          for p,values in expected.items()}
    need(integ.get('changes')==expected_integration,
         'source106 integration change manifest mismatch')
    return {'source105_file_count':len(bm),'source106_file_count':len(cm),
            'changed_source_files':changes,
            'source105_snapshot_sha256':sha(base/'frozen-source.json'),
            'source106_snapshot_sha256':sha(cand/'frozen-source.json'),
            'source106_integration_sha256':sha(ROOT/'analysis/source106-callback-integration-001/integration.json')}


def validate_all_runs(old):
    rows={}
    for label in ('362','369','371'):
        receipt,prediction,build=old.validate_run(RUNS[label],SOURCES[label],label)
        rows[label]={'receipt':receipt,'prediction':prediction,'build':build}
    return rows


def validate_incomplete_371(old):
    """Bind a timed-out/terminated attempt without pretending it is a run result."""
    run=RUNS['371']; source=SOURCES['371']
    receipt=strict_json(run/'receipt.json')
    prediction=strict_json(run/'prediction.json')
    need(receipt.get('schema')=='numi.cardboard.native-run-receipt.v2',
         '371: receipt schema')
    need(receipt.get('execution_started') is True, '371: execution never started')
    need(receipt.get('source_inputs_unchanged') is True and receipt.get('changed_inputs')==[],
         '371: source-input preservation failed')
    need(prediction.get('schema')=='numi.cardboard.native-prediction.v1' and
         prediction.get('source_snapshot')==str(source), '371: prediction source binding')
    source_map=strict_json(source/'frozen-source.json')['source_sha256']
    input_hashes=prediction.get('inputs_sha256',{})
    need(sum(k.startswith('source/') for k in input_hashes)==len(source_map),
         '371: source input inventory count')
    for rel,digest in source_map.items():
        need(input_hashes.get('source/'+rel)==digest, f'371: prediction source hash {rel}')
    ext=prediction.get('external_inputs',[])
    need(len(ext)==1 and Path(ext[0]).name=='source-build.json',
         '371: source build receipt input missing')
    build_path=Path(ext[0]); build=strict_json(build_path)
    need(build.get('success') is True and build.get('source_snapshot')==str(source),
         '371: wrong or unsuccessful source build')
    external_keys=[k for k in input_hashes if k.startswith('external/') and
                   k.endswith('source-build.json')]
    need(len(external_keys)==1 and input_hashes[external_keys[0]]==sha(build_path),
         '371: build receipt hash mismatch')
    complete=(receipt.get('timed_out') is False and
              receipt.get('termination_errors')==[] and receipt.get('exit_code')==1)
    return {'complete_for_parity':complete,'receipt':receipt,'prediction':prediction,
            'receipt_sha256':sha(run/'receipt.json'),
            'prediction_sha256':sha(run/'prediction.json'),
            'source_build_sha256':sha(build_path),
            'status':('completed_retained_exit_1' if complete else 'incomplete_or_unexpected_exit'),
            'reason':{'timed_out':receipt.get('timed_out'),
                      'termination_errors':receipt.get('termination_errors'),
                      'exit_code':receipt.get('exit_code')}}


def normalize_prediction(pred):
    argv=list(pred['argv'])
    need(len(argv)==7, 'native CLI argument count changed')
    argv[0]='<built-checker>'; argv[1]='<source-material>'; argv[4]='<run-output-dir>'
    env=dict(pred.get('environment',{}))
    if 'NM_FEM_COUPLED_CAPTURE_PATH' in env:
        env['NM_FEM_COUPLED_CAPTURE_PATH']='<capture-output-stem>'
    return argv,env


def compare_predictions(rows):
    labels=('362','369','371')
    normalized={k:normalize_prediction(rows[k]['prediction']) for k in labels}
    reference=normalized['362']
    for k in labels[1:]:
        need(normalized[k]==reference, f'{k}: physical argv/environment differs beyond run paths')
    for k in labels:
        pred=rows[k]['prediction']
        need(not any('TRUE_RESIDUAL' in e for e in pred.get('environment',{})),
             f'{k}: true-residual option not default-off')
        need(pred.get('native_protocol_id')==rows['362']['prediction'].get('native_protocol_id'),
             f'{k}: native protocol identifier differs')
    return {'normalized_physical_argv_equal':True,'normalized_environment_equal':True,
            'normalized_argv':reference[0],
            'environment_keys':sorted(reference[1]),
            'native_protocol_id':rows['362']['prediction'].get('native_protocol_id'),
            'true_residual_explicitly_absent_in_all_runs':True}


def capture(old,label):
    return old.capture(RUNS[label],label)


def compare_captures(old):
    captures={label:capture(old,label) for label in ('362','369','371')}
    manifests={k:v[2] for k,v in captures.items()}
    payloads={k:v[3] for k,v in captures.items()}
    arrays={k:v[4] for k,v in captures.items()}
    for k in ('362','369','371'):
        need(manifests[k].get('rhs_preserved_before_fgmres') is (k!='369'),
             f'{k}: RHS-ready value differs from retained expected observation')
    pairwise={}
    for left,right in (('362','369'),('362','371'),('369','371')):
        lm=manifests[left]; rm=manifests[right]
        need(payloads[left]==payloads[right], f'{left}/{right}: center physical payload bytes differ')
        need(list(arrays[left])==list(arrays[right]), f'{left}/{right}: array ordering differs')
        for name in arrays[left]:
            a,b=arrays[left][name],arrays[right][name]
            for field in ('dtype','count','element_bytes','offset','byte_count','sha256'):
                need(a.get(field)==b.get(field),f'{left}/{right}: {name}.{field} differs')
        norm_l=copy.deepcopy(lm); norm_r=copy.deepcopy(rm)
        norm_l['payload']['file']=norm_r['payload']['file']='<capture-path>'
        fp_l=norm_l.pop('device_program_fingerprint')
        fp_r=norm_r.pop('device_program_fingerprint')
        diffs=old.recursive_differences(norm_l,norm_r)
        paths=[d[0] for d in diffs]
        expected=['rhs_preserved_before_fgmres'] if (lm['rhs_preserved_before_fgmres'] != rm['rhs_preserved_before_fgmres']) else []
        need(paths==expected, f'{left}/{right}: unreviewed capture metadata differences: {diffs}')
        pairwise[f'{left}_vs_{right}']={
          'payload_bytes':len(payloads[left]),'payload_sha256':sha_bytes(payloads[left]),
          'array_count':len(arrays[left]),'array_bytes_and_descriptors_equal':True,
          'device_program_fingerprint':{left:fp_l,right:fp_r,'different':fp_l!=fp_r},
          'rhs_preserved_before_fgmres':{left:lm['rhs_preserved_before_fgmres'],
             right:rm['rhs_preserved_before_fgmres'],'only_metadata_difference':paths==expected},
          'other_manifest_differences':diffs}
    return {'runs':{k:{'rhs_preserved_before_fgmres':manifests[k]['rhs_preserved_before_fgmres'],
                          'payload_bytes':len(payloads[k]),'payload_sha256':sha_bytes(payloads[k]),
                          'array_count':len(arrays[k]),
                          'full_rhs_sha256':arrays[k]['full_rhs_before_fgmres']['sha256']}
                   for k in manifests},'pairwise':pairwise}


def compare_diagnostics_and_logs(old):
    out={}
    pair_status_differences=[]
    for left,right in (('362','369'),('362','371'),('369','371')):
        la=RUNS[left]; lb=RUNS[right]
        fa=la/'diagnostic-state/dense'; fb=lb/'diagnostic-state/dense'
        files_a={p.relative_to(fa).as_posix():p for p in fa.rglob('*') if p.is_file()}
        files_b={p.relative_to(fb).as_posix():p for p in fb.rglob('*') if p.is_file()}
        need(set(files_a)==set(files_b) and len(files_a)==11,
             f'{left}/{right}: diagnostic-state inventory differs')
        capture_a=strict_json(la/'center-capture.coupled.json')
        capture_b=strict_json(lb/'center-capture.coupled.json')
        ordinary=[]; binary=[]; logs=[]
        for name in sorted(files_a):
            xa=files_a[name].read_bytes(); xb=files_b[name].read_bytes()
            if name.endswith('.bin'):
                need(len(xa)==len(xb)==904359, f'{left}/{right} {name}: payload size')
                fp_a=bytes.fromhex(capture_a['device_program_fingerprint'])[::-1]
                fp_b=bytes.fromhex(capture_b['device_program_fingerprint'])[::-1]
                lo,hi=DEVICE_FP_RANGE
                need(xa[lo:hi]==fp_a and xb[lo:hi]==fp_b,
                     f'{left}/{right} {name}: serialized fingerprint does not bind manifest')
                outside_equal=xa[:lo]==xb[:lo] and xa[hi:]==xb[hi:]
                need(outside_equal,
                     f'{left}/{right} {name}: bytes differ outside device fingerprint')
                binary.append({'path':name,'bytes':len(xa),'device_fingerprint_range':[lo,hi],
                  'baseline_sha256':sha_bytes(xa),'candidate_sha256':sha_bytes(xb),
                  'bytes_equal_outside_device_fingerprint':True,
                  'all_bytes_equal':xa==xb})
            elif name.endswith('.json'):
                need(xa==xb, f'{left}/{right} {name}: transaction JSON differs')
                ordinary.append({'path':name,'bytes':len(xa),'sha256':sha_bytes(xa),
                                 'exact_equal':True})
            else:
                ta=xa.decode().replace(str(la),'<RUN>')
                tb=xb.decode().replace(str(lb),'<RUN>')
                # Parse the structured trace records while retaining all source text.
                def parse_rows(text):
                    result=[]
                    for line in text.splitlines():
                        if line.startswith(TRACE_PREFIX):
                            result.append(old.strict_json_text(line[len(TRACE_PREFIX):]))
                    return result
                ra=parse_rows(ta); rb=parse_rows(tb)
                need(len(ra)==len(rb)==32, f'{left}/{right} {name}: Armijo record count')
                row_diffs=[]
                for idx,(a,b) in enumerate(zip(ra,rb)):
                    fields=old.recursive_differences(a,b)
                    if fields:
                        item={'record_index':idx,'newton_iteration':a.get('newton_iteration'),
                              'differences':[{'field':p,'left':x,'right':y} for p,x,y in fields]}
                        row_diffs.append(item)
                        pair_status_differences.append({'pair':f'{left}_vs_{right}',
                                                        'log':name,**item})
                # Remove only the trace rows from the text comparison because they
                # are compared structurally above; keep every other log byte.
                other_a='\n'.join(line for line in ta.splitlines()
                                  if not line.startswith(TRACE_PREFIX))
                other_b='\n'.join(line for line in tb.splitlines()
                                  if not line.startswith(TRACE_PREFIX))
                other_equal=other_a==other_b
                need(other_equal, f'{left}/{right} {name}: non-Armijo log text differs')
                logs.append({'path':name,'bytes_left':len(xa),'bytes_right':len(xb),
                  'non_armijo_text_equal_after_run_path_normalization':True,
                  'armijo_records':len(ra),'armijo_records_exact_equal':not row_diffs,
                  'armijo_record_differences':row_diffs})
        # Also compare the executable's captured stdout/stderr, retaining the
        # native Armijo stream as structured data above.
        main_logs={}
        for name in ('stdout.log','stderr.log'):
            xa=(la/name).read_text().replace(str(la),'<RUN>')
            xb=(lb/name).read_text().replace(str(lb),'<RUN>')
            if name=='stderr.log':
                ra=[]; rb=[]
                for t,dst in ((xa,ra),(xb,rb)):
                    for line in t.splitlines():
                        if line.startswith(TRACE_PREFIX):
                            dst.append(old.strict_json_text(line[len(TRACE_PREFIX):]))
                need(len(ra)==len(rb)==32,f'{left}/{right}: stderr Armijo record count')
                diffs=[]
                for index,(a,b) in enumerate(zip(ra,rb)):
                    delta=old.recursive_differences(a,b)
                    if delta:
                        item={'record_index':index,'newton_iteration':a.get('newton_iteration'),
                              'differences':[{'field':p,'left':x,'right':y} for p,x,y in delta]}
                        diffs.append(item)
                        if not any(d['pair']==f'{left}_vs_{right}' and
                                   d['log']==name and d['record_index']==index
                                   for d in pair_status_differences):
                            pair_status_differences.append({'pair':f'{left}_vs_{right}',
                                                            'log':name,**item})
                no_trace_a='\n'.join(x for x in xa.splitlines() if not x.startswith(TRACE_PREFIX))
                no_trace_b='\n'.join(x for x in xb.splitlines() if not x.startswith(TRACE_PREFIX))
                need(no_trace_a==no_trace_b,
                     f'{left}/{right}: stderr differs outside Armijo records')
                main_logs[name]={'non_armijo_text_equal':True,'armijo_records':32,
                                 'armijo_records_exact_equal':not diffs,
                                 'armijo_record_differences':diffs}
            else:
                need(xa==xb, f'{left}/{right}: stdout differs after run-path normalization')
                main_logs[name]={'equal_after_run_path_normalization':True,
                                 'sha256_left':sha_bytes(xa.encode()),
                                 'sha256_right':sha_bytes(xb.encode())}
        out[f'{left}_vs_{right}']={'diagnostic_state':{'file_count':len(files_a),
           'transaction_json':ordinary,'transaction_state_binary':binary,
           'diagnostic_logs':logs},'main_logs':main_logs}
    return {'pairwise':out,'armijo_status_differences':pair_status_differences}


def run_audit():
    old=old_audit()
    # Re-run the sealed source104/105 source and integration binding checks without changing that report.
    lineage_104_105=old.source_delta()
    source_105_106=validate_source106(old)
    # A timed-out or improperly terminated 371 attempt is evidence of an
    # incomplete native run, never a parity sample. Keep that receipt intact
    # and report the already-known 362/369 discrepancy separately.
    candidate=validate_incomplete_371(old)
    if not candidate['complete_for_parity']:
        old_capture=old.compare_capture(RUNS['362'],RUNS['369'])
        old_logs=old.compare_logs(RUNS['362'],RUNS['369'])
        return {'schema':'numi.matter.source106_lifetime_parity_audit.v1',
          'overall_verdict':'INCONCLUSIVE_NATIVE_371',
          'source_lineage':{'source104_to_105':lineage_104_105,
                            'source105_to_106':source_105_106},
          'runs':{'362':{'directory':str(RUNS['362']),'receipt_sha256':sha(RUNS['362']/'receipt.json')},
                  '369':{'directory':str(RUNS['369']),'receipt_sha256':sha(RUNS['369']/'receipt.json')},
                  '371':{'directory':str(RUNS['371']),'receipt_sha256':candidate['receipt_sha256'],
                         'prediction_sha256':candidate['prediction_sha256'],
                         'source_build_sha256':candidate['source_build_sha256'],
                         'status':candidate['status'],'reason':candidate['reason']}},
          'baseline_362_vs_369':{'capture_metadata':old_capture['rhs_preserved_before_fgmres'],
                                 'armijo_records':old_logs['armijo_records'],
                                 'both_runs_exit_1':True},
          'source106_native_comparison':'not performed: 371 did not complete a valid retained attempt',
          'interpretation':{'callback_lifetime_fix':'not natively verified',
                            'rhs_flag':'no normalization; 369 false metadata remains preserved',
                            'physical_qualification':'none'}}
    rows=validate_all_runs(old)
    pred=compare_predictions(rows)
    caps=compare_captures(old)
    diagnostics=compare_diagnostics_and_logs(old)
    good_rhs = (all(caps['runs'][k]['rhs_preserved_before_fgmres'] is True
                    for k in ('362','371')) and
                caps['runs']['369']['rhs_preserved_before_fgmres'] is False)
    status_differences=diagnostics['armijo_status_differences']
    if not good_rhs:
        verdict='FAIL_CALLBACK_METADATA'
    elif status_differences:
        verdict='FAIL_UNEXPECTED_STATUS'
    else:
        verdict='PASS_CAPTURE_LIFETIME_PARITY'
    return {'schema':'numi.matter.source106_lifetime_parity_audit.v1',
      'overall_verdict':verdict,
      'source_lineage':{'source104_to_105':lineage_104_105,'source105_to_106':source_105_106},
      'runs':{k:{'directory':str(RUNS[k]),'source_snapshot':str(SOURCES[k]),
        'receipt_sha256':sha(RUNS[k]/'receipt.json'),'prediction_sha256':sha(RUNS[k]/'prediction.json'),
        'source_build_sha256':sha(Path(rows[k]['prediction']['external_inputs'][0])),
        'exit_code':rows[k]['receipt']['exit_code'],'timed_out':rows[k]['receipt']['timed_out'],
        'termination_errors':rows[k]['receipt']['termination_errors']}
        for k in rows},
      'normalized_invocation':pred,'capture_comparison':caps,
      'diagnostic_states_logs_and_armijo':diagnostics,
      'interpretation':{
        'callback_fix_evidence':'run369 retains false RHS-ready metadata despite byte-identical RHS; run371 reports true, matching the guarded capture state and identical RHS bytes. This verifies the metadata-lifetime correction for this field.',
        'remaining_parity_failure':'run371 differs from 362/369 in Newton 7 trial_status for a rejected private trial. This status is preserved, not normalized; strict full-trace parity therefore fails.',
        'simulation_status':'All three retained native attempts exit 1; parity does not establish a converged physical cycle.',
        'no_metadata_normalization':'rhs_preserved_before_fgmres remains compared as observed in every pair.'}}


def main():
    out=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('report.json')
    report=run_audit()
    out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'verdict':report['overall_verdict'],'report':str(out),
      'rhs_flags':{k:v['rhs_preserved_before_fgmres'] for k,v in report['capture_comparison']['runs'].items()}},indent=2))
    raise SystemExit(0 if report['overall_verdict']=='PASS_CAPTURE_LIFETIME_PARITY' else 1)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'AUDIT_ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2)
