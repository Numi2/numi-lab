#!/usr/bin/env python3
"""Preflight and source-prefix validator for a source167 resumed diagnostic segment.

This tool never launches the native executable. It does not inspect the source470
runtime outputs in its default preflight mode. `--check-prefix` is intended only
after an explicit terminal source470 receipt exists; it validates that accepted
checkpoint prefix before printing the exact source-bound wrapper argv.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, subprocess, sys

PACKAGE=Path(__file__).resolve().parent
SPEC=PACKAGE/'native-471-command.json'
MANIFEST=PACKAGE/'package-manifest.json'
MIB=1024**2
GIB=1024*MIB

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def parse_properties(path):
    values={}
    for line in Path(path).read_text().splitlines():
        k,sep,v=line.partition('=')
        if not sep or not k or k in values:
            raise ValueError(f'malformed or duplicate resume metadata property: {k!r}')
        values[k]=v
    return values

def verify_package(spec):
    manifest=json.loads(MANIFEST.read_text())
    if manifest.get('schema')!='numi.cardboard.native471-resume-command-package.v1':
        raise ValueError('unsupported 471 package manifest')
    listed=set(manifest['files_sha256'])
    actual={str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob('*')
            if p.is_file() and p!=MANIFEST and '__pycache__' not in p.parts}
    if listed!=actual: raise ValueError(f'package inventory mismatch: {sorted(actual-listed)} / {sorted(listed-actual)}')
    for rel,digest in manifest['files_sha256'].items():
        p=PACKAGE/rel
        if p.is_symlink() or not p.is_file() or sha(p)!=digest:
            raise ValueError(f'package file integrity failure: {rel}')
    source=Path(spec['source_snapshot']).resolve(strict=True)
    frozen_path=Path(spec['frozen_source_file']).resolve(strict=True)
    if sha(frozen_path)!=spec['frozen_source_sha256']:
        raise ValueError('frozen source manifest changed')
    frozen=json.loads(frozen_path.read_text())
    if len(frozen.get('source_sha256',{}))!=spec['source_file_count']:
        raise ValueError('source-file inventory count differs')
    for rel,digest in frozen['source_sha256'].items():
        p=(source/rel).resolve(strict=True)
        if not p.is_relative_to(source) or sha(p)!=digest: raise ValueError(f'source changed: {rel}')
    build=Path(spec['runtime_dir']).resolve(strict=True)
    prov=Path(spec['source_build_provenance']).resolve(strict=True)
    if sha(prov)!=spec['source_build_sha256'] or prov!=build/'source-build.json':
        raise ValueError('source build provenance changed')
    b=json.loads(prov.read_text())
    if not b.get('success') or b.get('source_sha256')!=frozen['source_sha256'] or b.get('targets')!=spec['source_build_targets'] or b.get('artifacts_sha256')!=spec['source_artifact_sha256']:
        raise ValueError('successful source167 build does not bind the frozen source/artifacts')
    for name,digest in b['artifacts_sha256'].items():
        p=Path(name).resolve(strict=True)
        if not p.is_relative_to(build) or sha(p)!=digest: raise ValueError(f'build artifact changed: {name}')
    for binding in spec['input_bindings']:
        if binding['kind']=='static_source_input':
            p=Path(binding['path']).resolve(strict=True)
            if sha(p)!=binding['sha256']: raise ValueError(f'static input changed: {binding["alias"]}')
            if binding.get('package_path'):
                cp=Path(binding['package_path'])
                if not cp.is_file() or sha(cp)!=binding['package_sha256']: raise ValueError(f'packaged input changed: {binding["alias"]}')
    case=Path(spec['case'])
    if case.exists() or case.is_symlink(): raise ValueError('471 case must be fresh')
    if case.parent.resolve()!=Path('/Users/home/cardboard-implementation-20261014/solver').resolve():
        raise ValueError('471 case is outside the bound solver directory')
    native=spec['native_argv']
    if native[native.index('--coupon-resume-checkpoint')+1]!=spec['checkpoint_contract']['checkpoint']:
        raise ValueError('checkpoint argv path differs from spec')
    if native[native.index('--coupon-resume-metadata')+1]!=spec['checkpoint_contract']['resume_metadata']:
        raise ValueError('metadata argv path differs from spec')
    if '--coupon-checkpoint-after-control' in native:
        raise ValueError('resume command also requests a new prefix stop')
    if native[native.index('--output')+1]!=str(case/'output'):
        raise ValueError('resumed output is not case-local')
    if spec['environment_overrides'].get('NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR')!='20:0:63:1:0:0':
        raise ValueError('consumer capture selector changed')
    if spec['checkpoint_contract']['source_control_index']!=19 or spec['checkpoint_contract']['next_control_index']!=20:
        raise ValueError('unexpected checkpoint cursor')
    base_cmd=json.loads(Path(spec['source470_command_path']).read_text())
    if sha(spec['source470_command_path'])!=spec['source470_command_sha256']:
        raise ValueError('source470 command lineage file changed')
    def physical_tokens(values, removable):
        out=[]; i=1
        while i<len(values):
            if values[i] in removable:
                i+=2
                continue
            out.append(values[i]); i+=1
        return out
    expected=physical_tokens(base_cmd['native_argv'], {'--metallib','--output','--coupon-checkpoint-after-control'})
    actual=physical_tokens(native, {'--metallib','--output','--coupon-resume-checkpoint','--coupon-resume-metadata'})
    if actual!=expected: raise ValueError('resumed physical/material argv differs from source470 prefix run')
    if spec['environment_overrides']!=base_cmd['environment_overrides']:
        raise ValueError('runtime controls differ from source470; only registered output-path relocation is permitted')
    wrapper=spec['wrapper_argv']; split=wrapper.index('--')
    if wrapper[split+1:]!=native: raise ValueError('wrapper native argv differs from the bound native command')
    flags={wrapper[i+1] for i,x in enumerate(wrapper[:split-1]) if x=='--input-file'}
    expected_paths={x.get('package_path',x['path']) for x in spec['input_bindings'] if x['kind']=='static_source_input'}|{x['path'] for x in spec['input_bindings'] if x['kind']=='source470_runtime_output'}
    if flags!=expected_paths: raise ValueError('wrapper input inventory differs from source/package bindings')
    wrapper_env={wrapper[i+1].split('=',1)[0]:wrapper[i+1].split('=',1)[1] for i,x in enumerate(wrapper[:split-1]) if x=='--env'}
    if wrapper_env!=spec['environment_overrides']: raise ValueError('wrapper runtime controls differ from bound settings')
    return {'source_files':len(frozen['source_sha256']), 'artifacts':len(b['artifacts_sha256']), 'static_inputs':sum(x['kind']=='static_source_input' for x in spec['input_bindings']), 'source_build_sha256':sha(prov), 'case_fresh':True, 'argv_matches_source470_physical_settings':True}

def verify_prefix(spec):
    """Validate source470 only when explicitly asked after its run is terminal."""
    root=Path(spec['source_output']).resolve(strict=True)
    case=Path(spec['source_case']).resolve(strict=True)
    cp=Path(spec['checkpoint_contract']['checkpoint'])
    meta=Path(spec['checkpoint_contract']['resume_metadata'])
    pref=Path(spec['checkpoint_contract']['prefix_manifest'])
    def regular(p,cap):
        if p.is_symlink() or not p.is_file() or p.resolve(strict=True)!=p.resolve(): raise ValueError(f'unsafe/missing source prefix file: {p}')
        st=p.stat()
        if st.st_nlink!=1 or not 0<st.st_size<=cap: raise ValueError(f'source prefix file is not bounded/unlinked: {p}')
        return st
    receipt=json.loads((case/'receipt.json').read_text())
    if receipt.get('schema')!='numi.cardboard.native-run-receipt.v2' or receipt.get('execution_started') is not True or receipt.get('exit_code')!=0 or receipt.get('timed_out') is not False or receipt.get('source_inputs_unchanged') is not True or receipt.get('changed_inputs')!=[]:
        raise ValueError('source470 receipt is not a successful unchanged terminal native run')
    result=json.loads((root/'result.json').read_text())
    if result.get('schema')!='numi.cardboard.probe-result.v1' or result.get('status')!='accepted_checkpoint_prefix_saved' or result.get('accepted_steps')!=20 or result.get('physical_validation') is not False:
        raise ValueError('source470 did not stop at an accepted incomplete control-19 prefix')
    report=json.loads((root/'coupon-checkpoint-prefix.json').read_text())
    if report.get('schema')!='numi.cardboard.coupon-checkpoint-prefix.v1' or report.get('status')!='accepted_checkpoint_prefix_saved' or report.get('accepted_control_index')!=19 or report.get('next_control_index')!=20 or any(report.get(k) is not False for k in ('full_protocol_complete','trajectory_parity_established','physical_validation')):
        raise ValueError('source470 checkpoint prefix report is not the expected incomplete boundary')
    if report.get('checkpoint_file')!='checkpoints/coupon-control-000019.nmcb' or report.get('metadata_file')!='checkpoints/coupon-control-000019.resume':
        raise ValueError('source470 checkpoint file names differ from source167 producer')
    cst=regular(cp,2*1024**3); mst=regular(meta,64*1024); pst=regular(pref,4*1024**2)
    if sha(cp)!=report.get('checkpoint_sha256') or sha(meta)!=report.get('metadata_sha256'):
        raise ValueError('source470 prefix report checkpoint/metadata hashes differ')
    m=parse_properties(meta)
    checks=[m.get('schema')=='numi.cardboard.accepted-coupon-resume.v2',m.get('accepted_control_index')=='19',m.get('next_control_index')=='20',m.get('source_prefix_control_count')=='20',m.get('environment_count')=='2',m.get('both_environments_accepted')=='true',m.get('checkpoint_sha256')==sha(cp),m.get('checkpoint_bytes')==str(cst.st_size),m.get('prefix_manifest_sha256')==sha(pref),m.get('prefix_manifest_file')!='',m.get('source_run_directory')==str(root)]
    if not all(checks):
        raise ValueError('source470 resume metadata does not certify expected checkpoint/cursor '+repr(checks))
    if report.get('metadata_sha256')!=sha(meta) or m.get('protocol_manifest_sha256')!=sha(root/'manifest.json'):
        raise ValueError('source470 protocol or metadata hash binding failed')
    if m.get('source_runner_case_directory')!=str(case):
        raise ValueError('checkpoint is not bound to source-bound runner case')
    # Verify prefix inventory line-by-line; this is the same semantics as the source C++ verifier.
    lines=pref.read_text().splitlines()
    fields={}; entries=[]
    for line in lines:
        if line.startswith('file='):
            parts=line[5:].split('|')
            if len(parts)!=4: raise ValueError('malformed prefix inventory entry')
            rel,n,digest,isobs=parts
            if rel.startswith('/') or '..' in Path(rel).parts or any(c in rel for c in ('\\','|','\n','\r')) or int(n)<=0 or not re.fullmatch(r'[0-9a-f]{64}',digest) or isobs not in ('0','1'):
                raise ValueError('unsafe prefix inventory entry')
            entries.append((rel,int(n),digest,isobs))
        else:
            k,sep,v=line.partition('=')
            if not sep or k in fields: raise ValueError('malformed/duplicate prefix header')
            fields[k]=v
    if fields.get('schema')!='numi.cardboard.accepted-coupon-prefix.v1' or fields.get('source_run_directory')!=str(root) or fields.get('accepted_control_index')!='19':
        raise ValueError('prefix manifest identity/cursor mismatch')
    obs=[e for e in entries if e[0]=='observations.csv' and e[3]=='1']
    if len({e[0] for e in entries})!=len(entries) or len(obs)!=1 or fields.get('observations_prefix_bytes')!=str(obs[0][1]) or fields.get('observations_prefix_sha256')!=obs[0][2]:
        raise ValueError('prefix file inventory is duplicate or has inconsistent observations binding')
    for rel,n,digest,_ in entries:
        p=root/rel; st=regular(p,2*1024**3)
        if st.st_size!=n or sha(p)!=digest: raise ValueError(f'accepted source prefix file mismatch: {rel}')
    if not (case/'diagnostic-captures.json').is_file(): raise ValueError('source470 capture-pending runner receipt is missing')
    cap=json.loads((case/'diagnostic-captures.json').read_text())
    pending=cap.get('pending_captures',[])
    if cap.get('status')!='validated_with_pending_checkpoint_targets' or not any(x.get('key')=='NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH' and x.get('target_control')==20 and x.get('accepted_control')==19 and x.get('status')=='not_yet_visited_at_accepted_checkpoint' and x.get('positive_qualification_allowed') is False for x in pending):
        raise ValueError('source470 capture was not recorded as a pending future target')
    return {'checkpoint_bytes':cst.st_size,'checkpoint_sha256':sha(cp),'metadata_sha256':sha(meta),'prefix_manifest_sha256':sha(pref),'prefix_entries':len(entries),'source_result_sha256':sha(root/'result.json'),'source_receipt_exit_code':receipt['exit_code'],'capture_pending':True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--check-prefix',action='store_true',help='only after source470 terminal receipt; validates its checkpoint without launching')
    ap.add_argument('--print-wrapper',action='store_true')
    args=ap.parse_args()
    spec=json.loads(SPEC.read_text())
    summary=verify_package(spec)
    if args.check_prefix: summary['source_prefix']=verify_prefix(spec)
    if args.print_wrapper and not args.check_prefix: raise SystemExit('--print-wrapper requires a successful --check-prefix')
    print(json.dumps({'status':'ready_for_root_launch' if args.check_prefix else 'static_preflight_passed_dynamic_checkpoint_not_inspected','validation':summary},sort_keys=True,indent=2))
    if args.print_wrapper:
        print('WRAPPER_ARGV_JSON='+json.dumps(spec['wrapper_argv']))

if __name__=='__main__': main()
