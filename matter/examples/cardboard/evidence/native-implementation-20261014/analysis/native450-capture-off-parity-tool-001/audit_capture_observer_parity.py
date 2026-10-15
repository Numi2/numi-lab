#!/usr/bin/env python3
"""Read-only attempted-history comparison for native450 vs capture-off452."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any

CAPTURE_KEYS = {'NM_FEM_BASE_CGLS_CAPTURE_NEWTON', 'NM_FEM_BASE_CGLS_CAPTURE_PATH'}
OMIT_PREFIXES = ('cardboard_step_timing=', 'fem_coupled_capture_',
                 'fem_base_cgls_capture=', 'fem_base_cgls_direction=')


def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def read_json(p: Path): return json.loads(p.read_text())


def command_parts(path: Path):
    cmd=read_json(path)
    sep=cmd.index('--')
    env={}
    inputs=[]
    for i,x in enumerate(cmd[:sep]):
        if x=='--env':
            k,v=cmd[i+1].split('=',1); env[k]=v
        elif x=='--input-file': inputs.append(cmd[i+1])
    return cmd,cmd[sep+1:],env,inputs


def normalize_probe_args(args: list[str], run_dir: Path) -> list[str]:
    out=[]; i=0
    while i<len(args):
        value=args[i]
        if value=='--output':
            if i+1>=len(args): raise ValueError('probe --output lacks a path')
            out.extend([value,'<run-output>']); i+=2; continue
        out.append(value.replace(str(run_dir),'<run-dir>'))
        i+=1
    return out


def validate_only_capture_env_removed(on_env: dict[str,str], off_env: dict[str,str],
                                      on_pred: dict[str,Any], off_pred: dict[str,Any],
                                      on_dir: Path, off_dir: Path) -> dict[str,Any]:
    on_keys,off_keys=set(on_env),set(off_env)
    removed=on_keys-off_keys; added=off_keys-on_keys
    if removed != CAPTURE_KEYS or added:
        raise ValueError(f'capture environment key delta is not exactly {sorted(CAPTURE_KEYS)}: removed={sorted(removed)}, added={sorted(added)}')
    mismatches=[]
    for k in sorted(off_keys):
        expected=on_env[k].replace(str(on_dir),str(off_dir))
        if off_env[k] != expected:
            mismatches.append({'key':k,'on_normalized':expected,'off':off_env[k]})
    if mismatches: raise ValueError(f'non-capture wrapper environment differs: {mismatches[:4]}')
    on_pe=on_pred.get('environment',{}); off_pe=off_pred.get('environment',{})
    if set(on_pe)-set(off_pe) != CAPTURE_KEYS or set(off_pe)-set(on_pe):
        raise ValueError('sealed prediction environments do not differ by exactly the two CGLS capture variables')
    for k in off_pe:
        expected=on_pe[k].replace(str(on_dir),str(off_dir)) if isinstance(on_pe[k],str) else on_pe[k]
        if off_pe[k] != expected: raise ValueError(f'prediction environment differs outside capture paths: {k}')
    return {'removed_keys_exact':True,'removed_keys':sorted(removed),'added_keys':sorted(added),
            'all_remaining_environment_values_equal_after_run_path_normalization':True,
            'prediction_environment_delta_exact':True}


def filtered_stderr(path: Path, run_dir: Path) -> list[str]:
    lines=[]
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith(OMIT_PREFIXES): continue
        lines.append(line.replace(str(run_dir),'<run-dir>'))
    return lines


def exact_tree(left: Path,right: Path) -> list[dict[str,Any]]:
    a={p.relative_to(left).as_posix():p for p in left.rglob('*') if p.is_file()}
    b={p.relative_to(right).as_posix():p for p in right.rglob('*') if p.is_file()}
    rows=[]
    for name in sorted(a.keys()|b.keys()):
        if name not in a or name not in b:
            rows.append({'path':name,'equal':False,'reason':'missing_on_'+('left' if name not in a else 'right')}); continue
        x=a[name].read_bytes(); y=b[name].read_bytes()
        rows.append({'path':name,'left_bytes':len(x),'right_bytes':len(y),
                     'left_sha256':hashlib.sha256(x).hexdigest(),'right_sha256':hashlib.sha256(y).hexdigest(),
                     'equal':x==y})
    return rows


def parse_prefixed(stderr: Path,prefix: str) -> list[str]:
    return [x for x in stderr.read_text(errors='replace').splitlines() if x.startswith(prefix)]


def verify(on_dir: Path, off_dir: Path, on_command: Path, off_command: Path) -> dict[str,Any]:
    on_dir=on_dir.resolve(); off_dir=off_dir.resolve()
    required=lambda d,n: d/n
    for d in (on_dir,off_dir):
        for name in ('prediction.json','receipt.json','stdout.log','stderr.log','output/manifest.json','output/compiled.nmatterpack','output/observations.csv'):
            if not required(d,name).is_file(): raise ValueError(f'run is incomplete: {d/name}')
    on_pred=read_json(on_dir/'prediction.json'); off_pred=read_json(off_dir/'prediction.json')
    on_rec=read_json(on_dir/'receipt.json'); off_rec=read_json(off_dir/'receipt.json')
    for label,rec in (('on',on_rec),('off',off_rec)):
        if rec.get('execution_started') is not True or rec.get('source_inputs_unchanged') is not True or rec.get('timed_out') is not False:
            raise ValueError(f'{label} receipt is incomplete, mutated, or timed out')
        if 'exit_code' not in rec: raise ValueError(f'{label} receipt lacks exit code')
    if on_pred.get('source_snapshot') != off_pred.get('source_snapshot') or on_pred.get('runtime_dir') != off_pred.get('runtime_dir'):
        raise ValueError('source/runtime snapshot differs between runs')
    if on_pred.get('executed_build_verified') is not True or off_pred.get('executed_build_verified') is not True:
        raise ValueError('one run lacks verified build provenance')
    if on_pred.get('build_provenance') != off_pred.get('build_provenance'):
        raise ValueError('build provenance path differs')
    on_argv,on_probe,on_env,on_inputs=command_parts(on_command)
    off_argv,off_probe,off_env,off_inputs=command_parts(off_command)
    if normalize_probe_args(on_probe,on_dir)!=normalize_probe_args(off_probe,off_dir):
        raise ValueError('physical probe arguments differ beyond the run-specific output path')
    if normalize_probe_args(on_pred.get('argv',[]),on_dir)!=normalize_probe_args(off_pred.get('argv',[]),off_dir):
        raise ValueError('sealed prediction probe arguments differ beyond the run-specific output path')
    if normalize_probe_args(on_pred.get('argv',[]),on_dir)!=normalize_probe_args(on_probe,on_dir) or \
            normalize_probe_args(off_pred.get('argv',[]),off_dir)!=normalize_probe_args(off_probe,off_dir):
        raise ValueError('sealed prediction argv does not match its prepared native command')
    env_check=validate_only_capture_env_removed(on_env,off_env,on_pred,off_pred,on_dir,off_dir)
    on_source={k:v for k,v in on_pred.get('inputs_sha256',{}).items() if k.startswith('source/')}
    off_source={k:v for k,v in off_pred.get('inputs_sha256',{}).items() if k.startswith('source/')}
    if on_source != off_source: raise ValueError('source input hashes differ')
    on_runtime={k:v for k,v in on_pred.get('inputs_sha256',{}).items() if k.startswith('runtime/')}
    off_runtime={k:v for k,v in off_pred.get('inputs_sha256',{}).items() if k.startswith('runtime/')}
    if on_runtime != off_runtime: raise ValueError('runtime input hashes differ')
    on_manifest=read_json(on_dir/'output/manifest.json'); off_manifest=read_json(off_dir/'output/manifest.json')
    if int(on_manifest['compiled_world_fingerprint']) != int(off_manifest['compiled_world_fingerprint']):
        raise ValueError('compiled world fingerprints differ')
    if sha(on_dir/'output/compiled.nmatterpack') != sha(off_dir/'output/compiled.nmatterpack'):
        raise ValueError('compiled package hashes differ')
    output_rows=exact_tree(on_dir/'output',off_dir/'output')
    arm_on=parse_prefixed(on_dir/'stderr.log','fem_residual_armijo_iteration=')
    arm_off=parse_prefixed(off_dir/'stderr.log','fem_residual_armijo_iteration=')
    true_on=parse_prefixed(on_dir/'stderr.log','fem_true_residual_cycle=')
    true_off=parse_prefixed(off_dir/'stderr.log','fem_true_residual_cycle=')
    clean_on=filtered_stderr(on_dir/'stderr.log',on_dir); clean_off=filtered_stderr(off_dir/'stderr.log',off_dir)
    # Diagnostic snapshots intentionally differ in existence: only the on run owns base-cgls.bin.
    on_cgls=on_dir/'base-cgls.bin'; off_cgls=off_dir/'base-cgls.bin'
    if not on_cgls.is_file() or off_cgls.exists(): raise ValueError('CGLS capture artifact presence does not match on/off selectors')
    on_status=on_dir/'first-status5.bin'; off_status=off_dir/'first-status5.bin'
    status5_present=on_status.is_file() and off_status.is_file()
    status5_equal=status5_present and on_status.read_bytes()==off_status.read_bytes()
    on_status_json=on_dir/'first-status5.json'; off_status_json=off_dir/'first-status5.json'
    status5_json_equal=False
    if on_status_json.is_file() and off_status_json.is_file():
        on_status_record=read_json(on_status_json); off_status_record=read_json(off_status_json)
        on_status_record['payload']['path']=on_status_record['payload']['path'].replace(str(on_dir),'<run-dir>')
        off_status_record['payload']['path']=off_status_record['payload']['path'].replace(str(off_dir),'<run-dir>')
        status5_json_equal=on_status_record==off_status_record
    on_result=json.loads((on_dir/'stdout.log').read_text().splitlines()[0])
    off_result=json.loads((off_dir/'stdout.log').read_text().splitlines()[0])
    history={
          'complete_receipts':True,'both_processes_reached_terminal_receipt':True,
          'same_exit_code':on_rec['exit_code']==off_rec['exit_code'],
          'stdout_byte_exact':(on_dir/'stdout.log').read_bytes()==(off_dir/'stdout.log').read_bytes(),
          'observations_byte_exact':(on_dir/'output/observations.csv').read_bytes()==(off_dir/'output/observations.csv').read_bytes(),
          'armijo_record_count_on':len(arm_on),'armijo_record_count_off':len(arm_off),'full_armijo_trace_byte_exact':arm_on==arm_off,
          'true_residual_record_count_on':len(true_on),'true_residual_record_count_off':len(true_off),'full_true_residual_trace_byte_exact':true_on==true_off,
          'remaining_stderr_lines_after_exclusions_on':len(clean_on),'remaining_stderr_lines_after_exclusions_off':len(clean_off),
          'remaining_stderr_byte_exact_after_run_path_normalization':clean_on==clean_off,
          'first_status5_payload_present_both':status5_present,'first_status5_payload_byte_exact':status5_equal,
          'first_status5_manifest_equal_after_run_path_normalization':status5_json_equal,
          'physical_output_file_count':len(output_rows),'physical_output_files_byte_exact':sum(bool(r['equal']) for r in output_rows),
          'output_file_differences':[r for r in output_rows if not r['equal']]}
    parity_pass=bool(history['same_exit_code'] and history['stdout_byte_exact'] and
        history['observations_byte_exact'] and history['full_armijo_trace_byte_exact'] and
        history['full_true_residual_trace_byte_exact'] and
        history['remaining_stderr_byte_exact_after_run_path_normalization'] and
        history['first_status5_payload_present_both'] and history['first_status5_payload_byte_exact'] and
        history['first_status5_manifest_equal_after_run_path_normalization'] and
        history['physical_output_files_byte_exact']==history['physical_output_file_count'])
    return {
      'schema':'numi.cardboard.native_cgls_capture_observer_parity.v1',
      'classification':'complete attempted-history observer parity only; no coupon completion or physical qualification',
      'parity_pass':parity_pass,
      'run_on':{'path':str(on_dir),'prediction_sha256':sha(on_dir/'prediction.json'),'receipt_sha256':sha(on_dir/'receipt.json'),
                'stderr_sha256':sha(on_dir/'stderr.log'),'stdout_sha256':sha(on_dir/'stdout.log'),
                'receipt_exit_code':on_rec['exit_code'],'receipt_timed_out':on_rec['timed_out'],
                'result':on_result,'CGLS_binary_sha256':sha(on_cgls)},
      'run_off':{'path':str(off_dir),'prediction_sha256':sha(off_dir/'prediction.json'),'receipt_sha256':sha(off_dir/'receipt.json'),
                 'stderr_sha256':sha(off_dir/'stderr.log'),'stdout_sha256':sha(off_dir/'stdout.log'),
                 'receipt_exit_code':off_rec['exit_code'],'receipt_timed_out':off_rec['timed_out'],'result':off_result,
                 'CGLS_binary_absent':not off_cgls.exists()},
      'command_comparison':{'probe_args_equal_except_output_path':True,'environment':env_check,
                            'on_command_sha256':sha(on_command),'off_command_sha256':sha(off_command),
                            'input_files_on':on_inputs,'input_files_off':off_inputs},
      'compiled_input_comparison':{'source_snapshot_equal':True,'runtime_dir_equal':True,
          'source_input_hashes_equal':True,'runtime_input_hashes_equal':True,
          'world_fingerprint_equal':True,'compiled_pack_sha256_equal':True,
          'world_fingerprint_hex':f"{int(on_manifest['compiled_world_fingerprint']):016x}"},
      'history_comparison':history,
      'excluded_observer_material':{'capture_only_stderr_prefixes':list(OMIT_PREFIXES),
          'CGLS_snapshot_absent_in_off_run':True,'status5_payload_compared':status5_present},
      'limits':['A pass means the capture observer did not change this attempted trajectory under these exact source/build/settings. It is not a solver or physical qualification.',
          'The comparison removes only CGLS capture markers, full-capture markers, and timing lines from stderr. Numerical Armijo and true-residual records remain in the comparison.',
          'The CGLS candidate direction/trace cannot transfer the original Newton certificate; final acceptance remains independently reported.']
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--on-run-dir',type=Path,required=True)
    ap.add_argument('--off-run-dir',type=Path,required=True)
    ap.add_argument('--on-command',type=Path,required=True)
    ap.add_argument('--off-command',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    out=args.output.resolve()
    if out.exists(): raise SystemExit(f'refusing to overwrite {out}')
    report=verify(args.on_run_dir,args.off_run_dir,args.on_command,args.off_command)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'result':'AUDIT_WRITTEN','output':str(out),'parity_pass':report['parity_pass'],
        'armijo_exact':report['history_comparison']['full_armijo_trace_byte_exact'],
        'true_residual_exact':report['history_comparison']['full_true_residual_trace_byte_exact'],
        'physical_outputs_exact':report['history_comparison']['physical_output_files_byte_exact']==report['history_comparison']['physical_output_file_count']},indent=2))
    if not report['parity_pass']: return 2

if __name__=='__main__': main()
