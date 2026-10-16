#!/usr/bin/env python3
"""Source-bound native469 runner v4. Default is CPU-only; native work needs --execute."""
from __future__ import annotations
import argparse, contextlib, fcntl, hashlib, json, os, pathlib, shutil, signal, subprocess, sys, tempfile, time, uuid

ROOT = pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
PKG = ROOT / 'analysis/native469-material-replay-174-host-macmini-002'
ARTIFACT_PKG = ROOT / 'analysis/native469-material-replay-174-host-macmini-001'
SRC = ROOT / 'snapshots/source-checkpoint-warm-breakdown-173'
BUILD = ROOT / 'build-source173-001'
BIN = ARTIFACT_PKG / 'tools/numi-matter-cardboard-source469-material-replay-check'
RUNS = PKG / 'runs/invocations'
LOCK = SRC / 'build-cardboard-native-worker/worker.lock'
DISK_FLOOR = 2 * 1024**3
MEMORY_LIMIT = 16 * 1024**3
LOG_LIMIT = 256 * 1024**2
INPUTS = [
 ('capture','inputs/capture/final-newton','cf42bdb145317b24a044e8e7561a99dc987df44761b841a08d7f3b28e6cd96d2'),
 ('compiled_package','inputs/origin/compiled.nmatterpack','7b4df5b67bacfd579e3ce4b6776457f581cfb3240cbc70552b5d5c716f987be1'),
 ('run_manifest','inputs/origin/run-manifest.json','142fa1f08cf14ea4eb0e77039174a2c25030e14c0d38d6c9874a06e561bd3f6f'),
 ('run_receipt','inputs/origin/run-receipt.json','35b856b4d94bcc3acfb5e611847ecc166860a228efc81e841decec019ea7e17f'),
 ('mp_report','inputs/mp/report.json','029c3b5efa8ee3459c5016283ddf0bf09f1204a9359722847bc77bd23ffbb960'),
 ('mp_arrays','inputs/mp/mixed_reference_arrays.npz','107e942cc2e131eed8f8db7759f8deb3e86fcdc42d8da249996daf51d6ed11ad'),
 ('mp_manifest','inputs/mp/package-manifest.json','f176666da43b50a441c038cbd88b5c9c804df32fa8ea9d46483a086ab5616c19'),
 ('selected_references','inputs/references/selected-reference-rows.jsonl','d60d48e42f29608dfd9a0a1c04bc771fb544460966f9f9008628732a9d23dc49'),
]
EXPECTED = {
 'source':'6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3',
 'build':'96396de7fa14197a6759583d8b7bb053c412e08f1051b947502c1cc38b185229',
 'archive':'65703d9d857ec15d2d1e8ae93cac605ef5f46964b5a1d836f1af1b0eaf916047',
 'executable':'5dc0b921b41527b973b498d682f84b010b6f41f8acf4c11b1eec4a7305eef3b2',
 'metallib':'3996691d7674278d3cfffd4d6444b2412948824b78a28437af2da3c322a4fecd',
 'wrapper':'69dc60e47321ccca5ab09e5133d9f08620aca13a0bf15395e349e411329a0475',
 'shader':'23f6f758cda7ccc0e0651e45a298644e6b71a3c25cd5ef2f7c9cbe4ceca40628',
 'helper':'865a7399791390c3a1d27f9e8d70bb0aa9a5f8e52ebec14cd350a0c3fe258b81',
}
WORKER_MARKERS = ('numi-matter-', 'sparse-scan-checker', 'source469-material-replay-check')

class GuardError(RuntimeError):
    pass

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def verify(path,want,label):
    path=pathlib.Path(path)
    if not path.is_file(): raise GuardError('missing {}: {}'.format(label,path))
    got=sha(path)
    if got!=want: raise GuardError('{} hash mismatch: expected {}, got {}'.format(label,want,got))
    return got

def new_run_namespace(run_root=RUNS, mode='preflight', token=None):
    """Create an O_EXCL per-invocation namespace; never reuses receipts or outputs."""
    root=pathlib.Path(run_root)
    root.mkdir(parents=True,exist_ok=True)
    if token is None: token='{}-{}'.format(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()),uuid.uuid4().hex)
    if not token or '/' in token or token in ('.','..'): raise ValueError('invalid run namespace token')
    path=root/token
    path.mkdir(mode=0o700,exist_ok=False)
    (path/'invocation.json').write_text(json.dumps({'schema':'numi.cardboard.native469.invocation.v4','run_id':token,'mode':mode,'created_unix_ns':time.time_ns()},indent=2)+'\n',encoding='utf-8')
    return path

def prepare_invocation(mode,run_root=RUNS):
    if mode not in ('preflight','execute'): raise ValueError('mode must be preflight or execute')
    return new_run_namespace(run_root=run_root,mode=mode)

def write_json_new(path,value):
    path=pathlib.Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,sort_keys=True); f.write('\n')

def command(mode,outdir):
    values=[(name,digest) for name,_,digest in INPUTS]
    args=[str(BIN),mode]
    flags=['--capture-sha256','--package-sha256','--manifest-sha256','--receipt-sha256','--mp-reference-sha256','--mp-arrays-sha256','--mp-package-manifest-sha256','--reference-export-sha256']
    for flag,(_,digest) in zip(flags,values): args.extend([flag,digest])
    args.extend(str(ARTIFACT_PKG/rel) for _,rel,_ in INPUTS)
    args.append(str(outdir))
    return args

def verify_bindings(output_path):
    verify(SRC/'frozen-source.json',EXPECTED['source'],'source173 frozen source')
    verify(BUILD/'source-build.json',EXPECTED['build'],'source173 build provenance')
    verify(BUILD/'libnumi_matter_compiler.a',EXPECTED['archive'],'source173 compiler archive')
    verify(BIN,EXPECTED['executable'],'checker executable')
    verify(ARTIFACT_PKG/'shaders/cardboard_bilinear15_capture_diagnostic.metallib',EXPECTED['metallib'],'checker metallib')
    verify(ARTIFACT_PKG/'candidate/matter/tools/cardboard_source469_material_replay_check.mm',EXPECTED['wrapper'],'source173-bound wrapper')
    verify(ARTIFACT_PKG/'candidate/matter/tools/cardboard_bilinear15_capture_diagnostic.metal',EXPECTED['shader'],'shader source')
    verify(ARTIFACT_PKG/'candidate/matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm',EXPECTED['helper'],'helper source')
    checked=[]
    for name,rel,digest in INPUTS:
        checked.append({'name':name,'path':str(ARTIFACT_PKG/rel),'sha256':verify(ARTIFACT_PKG/rel,digest,name)})
    if pathlib.Path(output_path).exists(): raise GuardError('output path must be fresh: {}'.format(output_path))
    free=shutil.disk_usage(PKG).free
    if free<DISK_FLOOR: raise GuardError('disk free {} below required {}'.format(free,DISK_FLOOR))
    return checked,free

def process_snapshot():
    with tempfile.TemporaryFile(mode='w+t') as listing:
        subprocess.run(['/bin/ps','-axo','pid=,command='],text=True,stdout=listing,stderr=listing,check=True)
        listing.seek(0)
        lines=listing.read().splitlines()
    found=[]
    for line in lines:
        bits=line.strip().split(None,1)
        if not bits or not bits[0].isdigit() or int(bits[0])==os.getpid(): continue
        cmd=bits[1] if len(bits)>1 else ''
        if any(marker in cmd for marker in WORKER_MARKERS): found.append({'pid':int(bits[0]),'command':cmd})
    return found

def lock_busy(lock_path=LOCK):
    """Non-owning lock probe, called only when this process holds no lease."""
    lock_path=pathlib.Path(lock_path)
    if not lock_path.exists(): return False
    with lock_path.open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return True
        fcntl.flock(f,fcntl.LOCK_UN)
    return False

@contextlib.contextmanager
def worker_lease(lock_path=LOCK):
    lock_path=pathlib.Path(lock_path)
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    f=lock_path.open('a+')
    try:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise GuardError('another native worker owns {}'.format(lock_path)) from exc
        yield f
    finally:
        try: fcntl.flock(f,fcntl.LOCK_UN)
        finally: f.close()

def reject_workers(snapshot):
    if snapshot: raise GuardError('another cardboard/native worker is active: '+' | '.join('{} {}'.format(x['pid'],x['command']) for x in snapshot))

def terminate_group(proc):
    try: os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait()

def run_file_backed(argv,cwd,stdout_path,stderr_path,timeout_seconds,monitor=None):
    """Child output goes directly to O_EXCL files; no child stdout/stderr pipes."""
    stdout_path=pathlib.Path(stdout_path); stderr_path=pathlib.Path(stderr_path)
    with stdout_path.open('xb') as out, stderr_path.open('xb') as err:
        proc=subprocess.Popen(argv,cwd=str(cwd),stdout=out,stderr=err,start_new_session=True)
        started=time.monotonic(); reason=None
        while proc.poll() is None:
            if monitor is not None:
                reason=monitor(proc,stdout_path,stderr_path)
                if reason:
                    terminate_group(proc); break
            if time.monotonic()-started>timeout_seconds:
                reason='timeout'; terminate_group(proc); break
            time.sleep(.25)
        rc=proc.wait()
    return rc,time.monotonic()-started,reason

def run_cpu_admission(run_dir,timeout_seconds=1200):
    run_dir=pathlib.Path(run_dir)
    outdir=run_dir/'cpu-admission-output'
    argv=command('--cpu-admission-469',outdir)
    write_json_new(run_dir/'cpu-admission-command.json',{'argv':argv,'cwd':str(ARTIFACT_PKG),'gpu_execution':False,'mode':'cpu_admission'})
    rc,wall,reason=run_file_backed(argv,ARTIFACT_PKG,run_dir/'cpu-admission.stdout.log',run_dir/'cpu-admission.stderr.log',timeout_seconds)
    if reason or rc: raise GuardError('CPU admission failed: exit={} reason={}; see per-run logs'.format(rc,reason))
    admission=json.loads((run_dir/'cpu-admission.stdout.log').read_text(encoding='utf-8'))
    write_json_new(run_dir/'cpu-admission-receipt.json',{'schema':'numi.cardboard.native469.cpu-admission.v4','exit_code':rc,'wall_seconds':wall,'gpu_execution':False,'admission':admission})
    return admission

def rss_group_bytes(pgid):
    with tempfile.TemporaryFile(mode='w+t') as listing:
        subprocess.run(['/bin/ps','-axo','pid=,pgid=,rss='],text=True,stdout=listing,stderr=listing,check=True)
        listing.seek(0)
        lines=listing.read().splitlines()
    total=0
    for line in lines:
        bits=line.split()
        if len(bits)==3 and bits[1].isdigit() and int(bits[1])==pgid: total+=int(bits[2])*1024
    return total

def native_monitor(proc,stdout_path,stderr_path):
    rss=rss_group_bytes(proc.pid)
    if rss>MEMORY_LIMIT: return 'process_group_rss_exceeded_{}'.format(MEMORY_LIMIT)
    free=shutil.disk_usage(PKG).free
    if free<DISK_FLOOR: return 'disk_free_below_{}'.format(DISK_FLOOR)
    if stdout_path.stat().st_size+stderr_path.stat().st_size>LOG_LIMIT: return 'native_log_bytes_exceeded_{}'.format(LOG_LIMIT)
    return None

def preflight_status(run_dir,checked,free,admission):
    workers=process_snapshot(); busy=lock_busy()
    result={'schema':'numi.cardboard.native469.guarded-run.v4-preflight','status':'input_and_cpu_admission_pass_no_native_dispatch','gpu_execution':False,'launch_allowed':not workers and not busy,'launch_block_reason':None if not workers and not busy else 'active cardboard/native worker or locked source173 worker lease','active_workers':workers,'source173_lock_busy':busy,'run_id':run_dir.name,'run_namespace':str(run_dir),'input_hashes':checked,'disk_free_bytes':free,'disk_free_decimal_gb':free/1e9,'required_disk_floor_bytes':DISK_FLOOR,'required_memory_limit_bytes':MEMORY_LIMIT,'memory_limit_semantics':'sampled aggregate RSS for native process group; not an OS reservation','cpu_admission':admission,'native_argv_template':command('--native-469',run_dir/'native-output')}
    write_json_new(run_dir/'preflight-summary.json',result)
    return result

def execute_native(run_dir,timeout_seconds,checked,free):
    run_dir=pathlib.Path(run_dir)
    workers_before=process_snapshot()
    reject_workers(workers_before)
    with worker_lease(LOCK) as lease:
        # Do not probe LOCK while holding our own flock: lease ownership itself is the exclusion check.
        workers_after=process_snapshot()
        reject_workers(workers_after)
        outdir=run_dir/'native-output'
        if outdir.exists(): raise GuardError('native output exists; per-invocation O_EXCL refusal')
        argv=command('--native-469',outdir)
        write_json_new(run_dir/'native-execution-command.json',{'argv':argv,'source_sha256':EXPECTED['source'],'build_sha256':EXPECTED['build'],'executable_sha256':EXPECTED['executable'],'input_hashes':checked,'disk_free_before_bytes':free,'memory_limit_bytes':MEMORY_LIMIT,'disk_floor_bytes':DISK_FLOOR,'log_limit_bytes':LOG_LIMIT,'timeout_seconds':timeout_seconds,'run_id':run_dir.name,'lease_acquired':True,'workers_before_lease':workers_before,'workers_after_lease':workers_after})
        rc,wall,reason=run_file_backed(argv,ARTIFACT_PKG,run_dir/'native.stdout.log',run_dir/'native.stderr.log',timeout_seconds,monitor=native_monitor)
        receipt={'schema':'numi.cardboard.native469.guarded-run.v4-execution','execution_started':True,'run_id':run_dir.name,'exit_code':rc,'limit_reason':reason,'wall_seconds':wall,'output_path':str(outdir),'output_exists':outdir.exists(),'gpu_execution':True,'material_qualification':False,'lease_released_after_receipt':True}
        write_json_new(run_dir/'native-execution-receipt.json',receipt)
        return receipt

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--execute',action='store_true',help='explicitly dispatch diagnostic Metal replay; default runs CPU admission only')
    ap.add_argument('--timeout-seconds',type=int,default=7200)
    a=ap.parse_args(argv)
    if a.timeout_seconds<1 or a.timeout_seconds>14400: ap.error('timeout must be 1..14400 seconds')
    mode='execute' if a.execute else 'preflight'
    run_dir=prepare_invocation(mode)
    try:
        cpu_out=run_dir/'cpu-admission-output'
        checked,free=verify_bindings(cpu_out)
        admission=run_cpu_admission(run_dir)
        if not a.execute:
            result=preflight_status(run_dir,checked,free,admission)
            print(json.dumps(result,indent=2,sort_keys=True))
            return 0
        # Recheck all source and input hashes immediately before lease/dispatch.
        checked,free=verify_bindings(run_dir/'native-output')
        try:
            receipt=execute_native(run_dir,a.timeout_seconds,checked,free)
        except GuardError as exc:
            refusal={'schema':'numi.cardboard.native469.guarded-run.v4-refusal','execution_started':False,'run_id':run_dir.name,'gpu_execution':False,'reason':str(exc),'input_hashes':checked,'disk_free_bytes':free}
            write_json_new(run_dir/'native-execution-refusal.json',refusal)
            print(json.dumps(refusal,indent=2,sort_keys=True),file=sys.stderr)
            return 2
        print(json.dumps(receipt,indent=2,sort_keys=True))
        return receipt['exit_code'] or (2 if receipt['limit_reason'] else 0)
    except Exception as exc:
        failure={'schema':'numi.cardboard.native469.guarded-run.v4-preflight-failure','run_id':run_dir.name,'gpu_execution':False,'error':str(exc)}
        write_json_new(run_dir/'preflight-failure.json',failure)
        print(json.dumps(failure,indent=2,sort_keys=True),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
