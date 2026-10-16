#!/opt/homebrew/Cellar/python@3.13/3.13.14_1/bin/python3.13
"""Fail-closed host runner for the source170 sparse scan checker.

Default: preflight only. Metal is dispatched only by explicit --execute.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, signal, subprocess, sys, time

PACKAGE=Path(__file__).resolve().parent
ROOT=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
SNAP=ROOT/'snapshots/source-checkpoint-warm-breakdown-173'
BUILD=ROOT/'build-source173-001'
EXE=PACKAGE/'validation/sparse-scan-checker'
LIB=PACKAGE/'NumiMatter-O3.metallib'
MANIFEST=PACKAGE/'package-manifest.json'
LOCK=PACKAGE/'executions/.single-worker.lock'
MIB=1024**2
MIN_FREE=2*1024**3
RESERVE=128*MIB
ABORT_FLOOR=512*MIB
TIMEOUT=900
WORKER_NAMES=('sparse-scan-checker','numi-matter-cardboard-probe','numi-matter-cardboard-paper-material-check','numi-matter-cardboard-warm-start-check','numi-matter-cardboard-shipping-snapshot-check','numi-matter-cardboard-coupon-resume-check')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def safe_file(path):
    p=Path(path)
    return p.is_file() and not p.is_symlink()

def free_bytes(path=PACKAGE):
    st=os.statvfs(path)
    return st.f_bavail*st.f_frsize

def static_files():
    rows=set()
    for p in PACKAGE.rglob('*'):
        if p.is_symlink(): raise RuntimeError(f'package symlink forbidden: {p}')
        if not p.is_file(): continue
        rel=p.relative_to(PACKAGE)
        if rel.as_posix()=='package-manifest.json' or rel.parts[0]=='executions' or '__pycache__' in rel.parts:
            continue
        rows.add(rel.as_posix())
    return rows

def verify():
    if not safe_file(MANIFEST): raise RuntimeError('package manifest missing/unsafe')
    m=json.loads(MANIFEST.read_text())
    if m.get('schema')!='numi.cardboard.sparse-scan-macmini-package.v1': raise RuntimeError('bad package schema')
    files=m.get('files',{})
    if static_files()!=set(files): raise RuntimeError('static package file set differs from manifest')
    for rel,row in files.items():
        p=PACKAGE/rel
        if not safe_file(p) or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            raise RuntimeError(f'package file integrity mismatch: {rel}')
    hb=json.loads((PACKAGE/'host-bindings.json').read_text())
    old=json.loads((PACKAGE/'lineage/native-170-command.json').read_text())
    ov=json.loads((PACKAGE/'lineage/source-overlay.json').read_text())
    if sha(PACKAGE/'lineage/native-170-command.json')!=hb['source170_original_command_sha256']:
        raise RuntimeError('source170 command lineage hash mismatch')
    if sha(PACKAGE/'lineage/source-overlay.json')!=hb['source170_overlay_manifest_sha256']:
        raise RuntimeError('source170 overlay manifest lineage hash mismatch')
    if sha(PACKAGE/'lineage/source-overlay.patch')!=hb['source170_overlay_patch_sha256']:
        raise RuntimeError('source170 patch lineage hash mismatch')
    if sha(PACKAGE/'source/cardboard_sparse_private_contact_scan_native_check.mm')!=hb['source170_checker_sha256']:
        raise RuntimeError('checker source differs from sealed source170')
    if old.get('source_checker_sha256')!=hb['source170_checker_sha256'] or old.get('source_metallib_sha256')!=hb['source170_o3_metallib_sha256']:
        raise RuntimeError('source170 original command does not bind checker inputs')
    row=next(x for x in ov['changed_files'] if x['path']=='matter/tools/cardboard_sparse_private_contact_scan_native_check.mm')
    if row['candidate_sha256']!=hb['source170_checker_sha256']:
        raise RuntimeError('source170 overlay candidate does not bind checker')
    if sha(LIB)!=hb['source170_o3_metallib_sha256']:
        raise RuntimeError('source170 O3 metallib hash mismatch')
    artifact=next(x for x in ov['validation_artifacts'] if x['path']=='validation/NumiMatter-O3-deps.metallib')
    if artifact['sha256']!=sha(LIB): raise RuntimeError('metallib differs from sealed source170 artifact')
    inventory=json.loads((PACKAGE/'inputs/gpu-case-inventory.json').read_text())
    if inventory.get('native_case_count')!=49 or inventory.get('total_environment_pair_slots')!=6366845 or len(inventory.get('cases',[]))!=49:
        raise RuntimeError('49-case inventory contract mismatch')
    if sha(SNAP/'frozen-source.json')!=hb['remote_frozen_source_json_sha256']:
        raise RuntimeError('remote source173 snapshot manifest changed')
    if sha(SNAP/'matter/include/numi/matter/shared.h')!=hb['remote_shared_header_sha256'] or hb['remote_shared_header_sha256']!=hb['source170_shared_header_source169_sha256']:
        raise RuntimeError('remote shared.h ABI differs from source169 base')
    if sha(BUILD/'source-build.json')!=hb['remote_source_build_json_sha256']:
        raise RuntimeError('remote source173 build metadata changed')
    build=json.loads((BUILD/'source-build.json').read_text())
    if not build.get('success') or build.get('source_snapshot')!=str(SNAP):
        raise RuntimeError('remote source173 build not successful/bound')
    for path,want in hb['remote_source_build_artifacts'].items():
        if not safe_file(path) or sha(path)!=want: raise RuntimeError(f'remote source173 artifact changed: {path}')
    compile_record=json.loads((PACKAGE/'compile-command.json').read_text())
    expected_macro='-DNUMI_MATTER_METALLIB=\"'+str(LIB)+'\"'
    argv=compile_record.get('argv',[])
    if compile_record.get('exit_code')!=0 or compile_record.get('output_binary')!=str(EXE) or compile_record.get('output_binary_sha256')!=m['files'].get('validation/sparse-scan-checker',{}).get('sha256'):
        raise RuntimeError('checker compile record does not bind fresh executable')
    if str(SNAP/'matter/include') not in argv or str(SNAP/'include') not in argv or expected_macro not in argv:
        raise RuntimeError('checker compile record does not bind source173 headers and absolute source170 metallib')
    old_manifest=json.loads((PACKAGE/'lineage/native-170-package-manifest.json').read_text())
    inv_row=old_manifest['files']['inputs/gpu-case-inventory.json']
    if sha(PACKAGE/'inputs/gpu-case-inventory.json')!=inv_row['sha256']:
        raise RuntimeError('49-case inventory differs from sealed source170 package')
    if ov.get('base_frozen_source_json_sha256')!=hb['source170_base_frozen_source_json_sha256']:
        raise RuntimeError('source170 base snapshot lineage mismatch')
    if not safe_file(EXE) or sha(EXE)!=m['files']['validation/sparse-scan-checker']['sha256']:
        raise RuntimeError('fresh host checker binary hash mismatch')
    return m,hb,inventory

def worker_relevant(name):
    return name in WORKER_NAMES or name.startswith('sparse-scan') or 'numi-matter-' in name

def processes():
    text=subprocess.check_output(['ps','-axo','pid=,ppid=,pgid=,stat=,comm='],text=True)
    rows=[]
    for line in text.splitlines():
        a=line.split(None,4)
        if len(a)!=5: continue
        try: pid,ppid,pgid=map(int,a[:3])
        except ValueError: continue
        state,name=a[3],Path(a[4]).name
        if state.startswith('Z'): continue
        if worker_relevant(name):
            rows.append({'pid':pid,'ppid':ppid,'pgid':pgid,'state':state,'name':name})
    return rows

def preflight():
    m,hb,inv=verify()
    active=processes()
    free=free_bytes()
    locked=LOCK.exists()
    return {'schema':'numi.cardboard.sparse-scan-macmini-preflight.v1','gpu_dispatch_occurred':False,
      'launch_allowed':not active and not locked and free>=MIN_FREE,'active_workers':active,'single_worker_lock_present':locked,'available_bytes':free,
      'minimum_available_bytes':MIN_FREE,'reserve_bytes':RESERVE,'abort_floor_bytes':ABORT_FLOOR,
      'native_gpu_case_count':inv['native_case_count'],'total_environment_pair_slots':inv['total_environment_pair_slots'],
      'checker_binary_sha256':m['files']['validation/sparse-scan-checker']['sha256'],
      'metallib_sha256':hb['source170_o3_metallib_sha256'],'source173_snapshot_sha256':hb['remote_frozen_source_json_sha256'],
      'source173_build_sha256':hb['remote_source_build_json_sha256']}

def write_reserve(path, byte_count=RESERVE):
    if byte_count<=0: raise ValueError('reserve size must be positive')
    chunk_size=min(MIB,byte_count); written=0; h=hashlib.sha256(); chunk=os.urandom(chunk_size)
    with Path(path).open('xb') as f:
        while written<byte_count:
            block=chunk[:min(chunk_size,byte_count-written)]
            f.write(block); h.update(block); written+=len(block)
        f.flush(); os.fsync(f.fileno()); st=os.fstat(f.fileno())
    allocated=st.st_blocks*512
    if st.st_size!=byte_count or allocated<byte_count: raise RuntimeError('reserve did not allocate requested bytes')
    return {'bytes':st.st_size,'allocated_bytes':allocated,'sha256':h.hexdigest(),'fsynced':True}

def execute():
    before=preflight()
    if not before['launch_allowed']: raise RuntimeError('preflight blocks execution: '+json.dumps(before,sort_keys=True))
    try: fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError: raise RuntimeError('single-worker lock already exists')
    os.write(fd,json.dumps({'pid':os.getpid(),'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}).encode()); os.fsync(fd); os.close(fd)
    active=processes()
    if active:
        LOCK.unlink(missing_ok=True)
        raise RuntimeError('another Matter/GPU worker appeared after lock acquisition: '+json.dumps(active,sort_keys=True))
    run=PACKAGE/'executions'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run.mkdir(parents=True,exist_ok=False)
    reserve_path=run/'reserve.bin'; stdout_path=run/'stdout.log'; stderr_path=run/'stderr.log'; receipt_path=run/'receipt.json'
    receipt={'schema':'numi.cardboard.sparse-scan-macmini-execution.v1','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preflight':before,'timed_out':False,'space_abort':False,'termination_error':None,'exit_code':None,'gpu_case_count':49}
    proc=None
    try:
        receipt['reserve']=write_reserve(reserve_path)
        if free_bytes()<ABORT_FLOOR: raise RuntimeError('free bytes fell below abort floor after reserve')
        with stdout_path.open('xb') as out,stderr_path.open('xb') as err:
            proc=subprocess.Popen([str(EXE)],cwd=str(PACKAGE),stdout=out,stderr=err,start_new_session=True)
            deadline=time.monotonic()+TIMEOUT
            while proc.poll() is None:
                if free_bytes()<ABORT_FLOOR:
                    receipt['space_abort']=True; receipt['termination_error']=terminate_owned(proc); break
                if time.monotonic()>=deadline:
                    receipt['timed_out']=True; receipt['termination_error']=terminate_owned(proc); break
                time.sleep(0.5)
            if proc.poll() is None: proc.wait()
            receipt['exit_code']=proc.returncode
    except BaseException as exc:
        receipt['runner_exception']=type(exc).__name__+': '+str(exc)
        if proc is not None and proc.poll() is None: receipt['termination_error']=terminate_owned(proc)
        if proc is not None and proc.poll() is not None: receipt['exit_code']=proc.returncode
        raise
    finally:
        try: reserve_path.unlink(missing_ok=True)
        finally: LOCK.unlink(missing_ok=True)
        receipt['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with receipt_path.open('w') as f: json.dump(receipt,f,sort_keys=True,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    return receipt

def terminate_owned(proc):
    error=None
    try: os.killpg(proc.pid,signal.SIGTERM)
    except OSError as e: error=f'SIGTERM:{type(e).__name__}:{e}'
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid,signal.SIGKILL)
        except OSError as e: error=(error+';' if error else '')+f'SIGKILL:{type(e).__name__}:{e}'
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: error=(error+';' if error else '')+'owned checker did not reap after SIGKILL'
    return error

def main():
    ap=argparse.ArgumentParser()
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--verify-only',action='store_true',help='verify package and bound source/build hashes; do not inspect/dispatch GPU')
    mode.add_argument('--execute',action='store_true',help='explicitly dispatch the 49-case Metal differential checker')
    args=ap.parse_args()
    if args.verify_only:
        m,hb,inv=verify(); result={'package_verified':True,'gpu_dispatch_occurred':False,'native_case_count':inv['native_case_count'],'binary_sha256':m['files']['validation/sparse-scan-checker']['sha256'],'source173_build_sha256':hb['remote_source_build_json_sha256']}
    elif args.execute: result=execute()
    else: result=preflight()
    print(json.dumps(result,sort_keys=True,indent=2))
    if args.execute and result.get('exit_code')!=0: return 1
    if not args.execute and not args.verify_only and not result.get('launch_allowed'): return 2
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e:
        print(json.dumps({'error':type(e).__name__,'message':str(e),'gpu_dispatch_occurred':False},sort_keys=True),file=sys.stderr)
        raise SystemExit(2)
