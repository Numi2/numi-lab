#!/usr/bin/env python3
"""Strict audit of source104 run362 versus source105 default-off run369.

This compares retained native artifacts only. It never executes Matter and it
never normalizes the rhs_preserved_before_fgmres metadata field.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path('/Users/home/cardboard-implementation-20261014')
BASE_SOURCE = ROOT / 'snapshots/source-contact-health-diagnostics-104'
CAND_SOURCE = ROOT / 'snapshots/source-true-residual-certification-105'
BASE_RUN = ROOT / 'native/health-and-vt-marker-firstfatal-362'
CAND_RUN = ROOT / 'native/true-residual-default-parity-369'
INTEGRATION = ROOT / 'analysis/source105-certification-integration-001/integration.json'
OVERLAY_MANIFEST = ROOT / 'analysis/true-residual-certification-overlay-105-v2/manifest-v2.json'
TRACE_PREFIX = 'fem_residual_armijo_iteration='
DEVICE_FP_RANGE = (12, 20)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def strict_json(path: Path):
    def reject(token):
        raise ValueError(f'non-standard JSON token {token} in {path}')
    return json.loads(path.read_text(), parse_constant=reject)


def need(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def recursive_differences(left, right, path=''):
    out=[]
    if type(left) is not type(right):
        return [(path, left, right)]
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            current=f'{path}.{key}' if path else key
            if key not in left or key not in right:
                out.append((current, left.get(key, '<missing>'), right.get(key, '<missing>')))
            else:
                out.extend(recursive_differences(left[key],right[key],current))
    elif isinstance(left,list):
        if len(left)!=len(right): out.append((path+'.length',len(left),len(right)))
        for i,(a,b) in enumerate(zip(left,right)):
            out.extend(recursive_differences(a,b,f'{path}[{i}]'))
    elif left != right:
        out.append((path,left,right))
    return out


def source_delta():
    old_manifest=strict_json(BASE_SOURCE/'frozen-source.json')
    new_manifest=strict_json(CAND_SOURCE/'frozen-source.json')
    old=old_manifest['source_sha256']; new=new_manifest['source_sha256']
    need(len(old)==778 and len(new)==780,'source selected-file counts changed')
    need(set(old) <= set(new),'source105 removed files from source104 inventory')
    # Independently validate every source entry against its frozen checkout.
    for snapshot, mapping in ((BASE_SOURCE,old),(CAND_SOURCE,new)):
        for relative,digest in mapping.items():
            path=snapshot/relative
            need(path.is_file() and sha(path)==digest,
                 f'frozen source hash mismatch: {snapshot}/{relative}')
    changes={p:{'source104':old.get(p),'source105':new.get(p)}
             for p in sorted(set(old)|set(new)) if old.get(p)!=new.get(p)}
    expected={
        'matter/CMakeLists.txt':{
            'source104':'3d7c54b440f76bd70a600ecfcc2e8e27b82554fa437e25744e8e410121c4f868',
            'source105':'031c7f9a497a245aa222d832918812ba561673ec6774925ca0cfb49198425bea'},
        'matter/src/metal/fgmres.metalinc':{
            'source104':'eef8a0919636c513716528dcf04cf9d175801c25e7159125808cbb59a9b696d6',
            'source105':'1c423e02e189df17c925e0b77ded87465098278f836011c16aad1bb8f92bcec0'},
        'matter/src/runtime.mm':{
            'source104':'119148dedc593742dfe7f62c1b06c99e488b6c40a783aa839722f70d2da59c20',
            'source105':'924669986619a88c6e34ec79e306495c1113e6ea35b0c793e8a36cfede887ec2'},
        'matter/tools/fgmres_true_residual_native_check.mm':{
            'source104':None,'source105':'b854f1a83d3851e99f5648478e208e7236751d1728a51286bcecfa8d5466b589'},
        'matter/tools/test_true_residual_certification_contract.py':{
            'source104':None,'source105':'b03c7aa4cb8bb275d232a172dab662ef3be61762b31c95f4f1e084d087568b2d'},
    }
    need(changes==expected,f'source104/105 delta differs from exact five-file overlay: {changes}')
    overlay=strict_json(OVERLAY_MANIFEST)
    need(overlay.get('base_snapshot')==str(BASE_SOURCE),'105 overlay has wrong base')
    need(overlay.get('patch_sha256')=='5c519fb8e86e884616aad687be1009b939c162e2d374e3f295c0d9ed1b8b68dc',
         '105 overlay patch binding differs')
    integration=strict_json(INTEGRATION)
    need(integration.get('snapshot')==str(CAND_SOURCE) and
         integration.get('base_snapshot')==str(BASE_SOURCE),
         '105 integration source lineage mismatch')
    need(integration.get('overlay_manifest_sha256')==sha(OVERLAY_MANIFEST),
         'integration overlay-manifest hash mismatch')
    return {
        'source104_file_count':len(old),'source105_file_count':len(new),
        'changed_source_files':changes,
        'source104_snapshot_manifest_sha256':sha(BASE_SOURCE/'frozen-source.json'),
        'source105_snapshot_manifest_sha256':sha(CAND_SOURCE/'frozen-source.json'),
        'source105_integration_sha256':sha(INTEGRATION),
        'source105_overlay_manifest_sha256':sha(OVERLAY_MANIFEST),
        'source105_patch_sha256':overlay['patch_sha256'],
    }


def validate_run(run: Path, source: Path, label: str):
    receipt=strict_json(run/'receipt.json')
    prediction=strict_json(run/'prediction.json')
    need(receipt.get('schema')=='numi.cardboard.native-run-receipt.v2',f'{label}: receipt schema')
    need(receipt.get('execution_started') is True,f'{label}: execution did not start')
    need(receipt.get('source_inputs_unchanged') is True and receipt.get('changed_inputs')==[],
         f'{label}: input mutation detected')
    need(receipt.get('timed_out') is False and receipt.get('termination_errors')==[],
         f'{label}: timeout/termination error')
    need(receipt.get('exit_code')==1,f'{label}: retained failure status changed')
    need(prediction.get('schema')=='numi.cardboard.native-prediction.v1',f'{label}: prediction schema')
    need(prediction.get('source_snapshot')==str(source),f'{label}: wrong source snapshot')
    argv=prediction.get('argv'); env=prediction.get('environment',{})
    need(isinstance(argv,list) and argv[0].endswith('/numi-matter-cardboard-full-contact-armijo-check'),
         f'{label}: executable identity')
    need(len(argv)==7 and argv[2:]==['--sparse-free-body-parity','--diagnostic-state-output',
         str(run/'diagnostic-state'),'--extended-budget','--compensated-material-arithmetic'],
         f'{label}: physical CLI differs or malformed')
    expected_material=source/'matter/materials/nagasawa2013_medium_orthotropic.nmatter'
    need(Path(argv[1]).resolve()==expected_material.resolve() and expected_material.is_file(),
         f'{label}: source-bound material input')
    need(env.get('NM_FEM_COUPLED_CAPTURE_SELECTOR')=='1201:7',f'{label}: center capture selector')
    need(env.get('NM_FEM_COUPLED_CAPTURE_PATH')==str(run/'center-capture'),
         f'{label}: capture path binding')
    need(not any('TRUE_RESIDUAL' in key for key in env),
         f'{label}: true-residual option explicitly enabled')
    input_hashes=prediction.get('inputs_sha256',{})
    source_map=strict_json(source/'frozen-source.json')['source_sha256']
    need(sum(k.startswith('source/') for k in input_hashes)==len(source_map),
         f'{label}: source input inventory count')
    for rel,digest in source_map.items():
        need(input_hashes.get('source/'+rel)==digest,f'{label}: prediction source hash {rel}')
    ext=prediction.get('external_inputs',[])
    need(len(ext)==1 and Path(ext[0]).name=='source-build.json',f'{label}: source build receipt input')
    build_path=Path(ext[0])
    build=strict_json(build_path)
    need(build.get('success') is True and build.get('source_snapshot')==str(source),
         f'{label}: unsuccessful/wrong source build')
    external_keys=[k for k in input_hashes if k.startswith('external/') and k.endswith('source-build.json')]
    need(len(external_keys)==1 and input_hashes[external_keys[0]]==sha(build_path),
         f'{label}: prediction source build receipt hash')
    artifacts=build.get('artifacts_sha256',{})
    for relative in ('runtime/lib/libmetalrobo.dylib',
                     'runtime/matter/numi-matter-cardboard-full-contact-armijo-check',
                     'runtime/matter/shaders/NumiMatter.metallib'):
        source_key=relative
        artifact_name=Path(relative).name
        matches=[digest for path,digest in artifacts.items() if Path(path).name==artifact_name]
        need(len(matches)==1 and input_hashes.get(source_key)==matches[0],
             f'{label}: runtime artifact not bound to successful build: {relative}')
    return receipt,prediction,build


def normalized_argv(prediction):
    argv=list(prediction['argv'])
    argv[0]='<built-checker>'
    argv[1]='<source104-or-105-medium-material>'
    argv[4]='<run-specific-diagnostic-state-dir>'
    return argv


def compare_predictions(base,cand):
    need(normalized_argv(base)==normalized_argv(cand),'physical argv differs after path normalization')
    need(base.get('native_protocol_id')==cand.get('native_protocol_id'),'native protocol ID differs')
    need(base.get('environment',{}).keys()==cand.get('environment',{}).keys(),
         'environment key set differs')
    a=dict(base['environment']); b=dict(cand['environment'])
    a['NM_FEM_COUPLED_CAPTURE_PATH']=b['NM_FEM_COUPLED_CAPTURE_PATH']='<run-specific-capture-stem>'
    need(a==b,'environment differs outside the output capture path')
    base_inputs=base['inputs_sha256']; cand_inputs=cand['inputs_sha256']
    need(set(base_inputs) <= set(cand_inputs),
         'candidate prediction removed baseline input keys')
    changed={k for k in set(base_inputs)&set(cand_inputs) if base_inputs[k]!=cand_inputs[k]}
    expected_source={'source/matter/CMakeLists.txt','source/matter/src/metal/fgmres.metalinc',
                     'source/matter/src/runtime.mm'}
    expected_runtime={'runtime/lib/libmetalrobo.dylib',
                      'runtime/matter/numi-matter-cardboard-full-contact-armijo-check',
                      'runtime/matter/shaders/NumiMatter.metallib'}
    expected_external={k for k in cand_inputs if k.startswith('external/') and k.endswith('source-build.json')}
    need(changed==expected_source|expected_runtime|expected_external,
         f'run input hash deltas unexpected: {sorted(changed)}')
    added={k for k in cand_inputs if k not in base_inputs}
    need(added=={'source/matter/tools/fgmres_true_residual_native_check.mm',
                 'source/matter/tools/test_true_residual_certification_contract.py'},
         f'candidate input inventory added unexpected entries: {sorted(added)}')
    return {'normalized_physical_argv_equal':True,'normalized_environment_equal':True,
            'changed_shared_input_paths':sorted(changed),'added_source_input_paths':sorted(added)}


def capture(run: Path, label: str):
    manifest_path=run/'center-capture.coupled.json'; payload_path=run/'center-capture.coupled.bin'
    m=strict_json(manifest_path); raw=payload_path.read_bytes()
    need(m.get('schema')=='numi.matter.coupled_newton_capture.v1',f'{label}: capture schema')
    need(m.get('root')==1201 and m.get('iteration')==7 and m.get('environment_count')==1,
         f'{label}: selected identity')
    need(m.get('capture_point')=='after_fgmres_measure_and_human_direction_map_before_armijo_trial',
         f'{label}: capture point')
    need(m.get('payload',{}).get('file')==str(payload_path),f'{label}: payload path')
    need(len(raw)==m['payload']['byte_count'] and sha_bytes(raw)==m['payload']['sha256'],
         f'{label}: payload size/hash')
    arrays=m.get('arrays',[])
    need(len(arrays)==m.get('array_count')==145,f'{label}: array count')
    last=0
    by_name={}
    for a in arrays:
        start=int(a['offset']); size=int(a['byte_count']); end=start+size
        need(start>=last and size>=0 and end<=len(raw),f'{label}: invalid/overlap extent {a.get("name")}')
        data=raw[start:end]
        need(sha_bytes(data)==a['sha256'],f'{label}: array hash {a.get("name")}')
        need(a['name'] not in by_name,f'{label}: duplicate array name')
        by_name[a['name']]=a
        last=end
    need('full_rhs_before_fgmres' in by_name,f'{label}: full RHS missing')
    return manifest_path,payload_path,m,raw,by_name


def compare_capture(base_run:Path,cand_run:Path):
    bp,bb,bm,br,ba=capture(base_run,'362')
    cp,cb,cm,cr,ca=capture(cand_run,'369')
    need(br==cr,'center raw physical payload bytes differ')
    need(list(ba)==list(ca),'center array order differs')
    per_array=[]
    for name in ba:
        x,y=ba[name],ca[name]
        for field in ('dtype','count','element_bytes','offset','byte_count','sha256'):
            need(x.get(field)==y.get(field),f'array descriptor differs: {name}.{field}')
        per_array.append({'name':name,'byte_count':x['byte_count'],'sha256':x['sha256'],'equal':True})
    left=copy.deepcopy(bm); right=copy.deepcopy(cm)
    # Only path and compiled-program fingerprint are known provenance deltas.
    left['payload']['file']=right['payload']['file']='<run-specific-center-payload>'
    fp_left=left.pop('device_program_fingerprint')
    fp_right=right.pop('device_program_fingerprint')
    diffs=recursive_differences(left,right)
    diffpaths=[x[0] for x in diffs]
    need(diffpaths==['rhs_preserved_before_fgmres'],
         f'center manifest has unreviewed metadata changes: {diffs}')
    return {
        'payload_byte_count':len(br),'payload_sha256':sha_bytes(br),
        'array_count':len(per_array),'all_array_descriptors_and_bytes_equal':True,
        'full_rhs_sha256_equal':ba['full_rhs_before_fgmres']['sha256']==ca['full_rhs_before_fgmres']['sha256'],
        'source_physics_fingerprint_equal':bm['source_physics_fingerprint']==cm['source_physics_fingerprint'],
        'device_program_fingerprint':{'362':fp_left,'369':fp_right,'different_as_expected_source_build_provenance':fp_left!=fp_right},
        'expected_capture_path_difference_normalized':True,
        'unexpected_manifest_differences':[{'field':p,'362':a,'369':b} for p,a,b in diffs],
        'rhs_preserved_before_fgmres':{'362':bm['rhs_preserved_before_fgmres'],
                                      '369':cm['rhs_preserved_before_fgmres'],
                                      'equal':bm['rhs_preserved_before_fgmres']==cm['rhs_preserved_before_fgmres']}
    }


def trace_rows(run:Path):
    rows=[]
    for line in (run/'stderr.log').read_text().splitlines():
        if line.startswith(TRACE_PREFIX):
            rows.append(strict_json_text(line[len(TRACE_PREFIX):]))
    return rows


def strict_json_text(text):
    def reject(token): raise ValueError(f'non-standard JSON token {token}')
    return json.loads(text,parse_constant=reject)


def normalize_run_path(text,run):
    return text.replace(str(run),'<RUN>')


def compare_diagnostic_state(base_run:Path,cand_run:Path):
    a=base_run/'diagnostic-state/dense'; b=cand_run/'diagnostic-state/dense'
    files_a={p.relative_to(a).as_posix():p for p in a.rglob('*') if p.is_file()}
    files_b={p.relative_to(b).as_posix():p for p in b.rglob('*') if p.is_file()}
    need(set(files_a)==set(files_b) and len(files_a)==11,'diagnostic-state file inventory differs')
    ordinary=[]; state_bin=[]; logs=[]
    for name in sorted(files_a):
        x=files_a[name].read_bytes(); y=files_b[name].read_bytes()
        if name.endswith('.bin'):
            need(len(x)==len(y)==904359,f'{name}: binary snapshot length')
            base_fp=bytes.fromhex(strict_json(base_run/'center-capture.coupled.json')['device_program_fingerprint'])[::-1]
            cand_fp=bytes.fromhex(strict_json(cand_run/'center-capture.coupled.json')['device_program_fingerprint'])[::-1]
            lo,hi=DEVICE_FP_RANGE
            need(x[lo:hi]==base_fp and y[lo:hi]==cand_fp,f'{name}: FP word does not bind run fingerprint')
            need(x[:lo]==y[:lo] and x[hi:]==y[hi:],f'{name}: state bytes differ outside device fingerprint')
            state_bin.append({'path':name,'bytes':len(x),'different_byte_range':[lo,hi],
                              'baseline_sha256':sha_bytes(x),'candidate_sha256':sha_bytes(y),
                              'all_bytes_outside_device_program_fingerprint_equal':True})
        elif name.endswith('.json'):
            need(x==y,f'{name}: transaction JSON differs')
            ordinary.append({'path':name,'bytes':len(x),'sha256':sha_bytes(x),'exact_equal':True})
        else:
            tx=normalize_run_path(x.decode(),base_run)
            ty=normalize_run_path(y.decode(),cand_run)
            need(tx==ty,f'{name}: log differs beyond run-specific output paths')
            logs.append({'path':name,'bytes_362':len(x),'bytes_369':len(y),
                         'equal_after_run_path_normalization':True,
                         'baseline_sha256':sha_bytes(x),'candidate_sha256':sha_bytes(y)})
    return {'file_count':len(files_a),'transaction_json':ordinary,
            'state_binary':state_bin,'control_logs':logs}


def compare_logs(base_run,cand_run):
    stdout_a=(base_run/'stdout.log').read_text(); stdout_b=(cand_run/'stdout.log').read_text()
    need(stdout_a==stdout_b,'stdout differs')
    stderr_a=(base_run/'stderr.log').read_text(); stderr_b=(cand_run/'stderr.log').read_text()
    norm_a=normalize_run_path(stderr_a,base_run); norm_b=normalize_run_path(stderr_b,cand_run)
    need(norm_a==norm_b,'stderr differs beyond run-specific path')
    arows=trace_rows(base_run); brows=trace_rows(cand_run)
    need(len(arows)==len(brows)==32,'Armijo trace record count mismatch')
    need(arows==brows,'Armijo rows differ')
    return {'stdout_sha256':sha_bytes(stdout_a.encode()),'stdout_equal':True,
            'stderr_equal_after_run_path_normalization':True,
            'armijo_records':len(arows),'armijo_json_records_exact_equal':True,
            'exit_code_362':1,'exit_code_369':1,
            'physical_simulation_verdict':'both retained runs failed with exit 1; this audit does not claim a successful physical cycle'}


def run_audit():
    source=source_delta()
    rb,pb,buildb=validate_run(BASE_RUN,BASE_SOURCE,'362')
    rc,pc,buildc=validate_run(CAND_RUN,CAND_SOURCE,'369')
    prediction=compare_predictions(pb,pc)
    cap=compare_capture(BASE_RUN,CAND_RUN)
    ds=compare_diagnostic_state(BASE_RUN,CAND_RUN)
    logs=compare_logs(BASE_RUN,CAND_RUN)
    metadata_fail=cap['rhs_preserved_before_fgmres']['equal'] is False
    need(metadata_fail,'expected unexpected RHS metadata mismatch disappeared')
    return {
      'schema':'numi.matter.native_default_parity_audit.v1',
      'overall_verdict':'FAIL_UNEXPECTED_METADATA',
      'source':source,
      'runs':{
        '362':{'directory':str(BASE_RUN),'receipt_sha256':sha(BASE_RUN/'receipt.json'),
               'prediction_sha256':sha(BASE_RUN/'prediction.json'),'receipt':rb,
               'source_build_sha256':sha(Path(pb['external_inputs'][0]))},
        '369':{'directory':str(CAND_RUN),'receipt_sha256':sha(CAND_RUN/'receipt.json'),
               'prediction_sha256':sha(CAND_RUN/'prediction.json'),'receipt':rc,
               'source_build_sha256':sha(Path(pc['external_inputs'][0]))}},
      'normalized_prediction_comparison':prediction,
      'center_capture':cap,
      'diagnostic_state':ds,
      'native_logs_and_attempt_outcome':logs,
      'classification':{
        'physical_payload_parity':'PASS_WITH_DEVICE_PROGRAM_FINGERPRINT_MASK',
        'center_payload_and_all_145_arrays':'byte-identical',
        'transaction_state_payloads':'identical outside the serialized device_program_fingerprint word at bytes 12..19',
        'transaction_metadata':'exactly equal',
        'armijo_trace':'32 exact records',
        'unexpected_metadata':'rhs_preserved_before_fgmres true in 362, false in 369 despite identical full_rhs_before_fgmres payload',
        'interpretation':'the 369 result is not a strict parity pass; preserve the false bit as an instrumentation discrepancy and verify with a fresh callback-lifetime capture',
        'physical_qualification':'none; both native attempts exit 1'
      }
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('report.json'))
    args=parser.parse_args()
    report=run_audit()
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'verdict':report['overall_verdict'],
                      'physical_payload_parity':report['classification']['physical_payload_parity'],
                      'rhs_flags':report['center_capture']['rhs_preserved_before_fgmres'],
                      'report':str(args.output)},indent=2))
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(f'AUDIT_ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2)
