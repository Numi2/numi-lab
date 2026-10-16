#!/usr/bin/env python3
"""Source-bound native469 wrapper. Default: hashes + CPU admission only; GPU requires --execute."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, pathlib, shutil, signal, subprocess, sys, time
ROOT=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
PKG=ROOT/'analysis/native469-material-replay-174-host-macmini-001'
SRC=ROOT/'snapshots/source-checkpoint-warm-breakdown-173'
BUILD=ROOT/'build-source173-001'
BIN=PKG/'tools/numi-matter-cardboard-source469-material-replay-check'
OUT=PKG/'runs/native469-source173-material-replay/output'
LOCK=SRC/'build-cardboard-native-worker/worker.lock'
DISK_FLOOR=2*1024**3
MEMORY_LIMIT=16*1024**3
INPUTS=[
 ('capture','inputs/capture/final-newton','cf42bdb145317b24a044e8e7561a99dc987df44761b841a08d7f3b28e6cd96d2'),
 ('compiled_package','inputs/origin/compiled.nmatterpack','7b4df5b67bacfd579e3ce4b6776457f581cfb3240cbc70552b5d5c716f987be1'),
 ('run_manifest','inputs/origin/run-manifest.json','142fa1f08cf14ea4eb0e77039174a2c25030e14c0d38d6c9874a06e561bd3f6f'),
 ('run_receipt','inputs/origin/run-receipt.json','35b856b4d94bcc3acfb5e611847ecc166860a228efc81e841decec019ea7e17f'),
 ('mp_report','inputs/mp/report.json','029c3b5efa8ee3459c5016283ddf0bf09f1204a9359722847bc77bd23ffbb960'),
 ('mp_arrays','inputs/mp/mixed_reference_arrays.npz','107e942cc2e131eed8f8db7759f8deb3e86fcdc42d8da249996daf51d6ed11ad'),
 ('mp_manifest','inputs/mp/package-manifest.json','f176666da43b50a441c038cbd88b5c9c804df32fa8ea9d46483a086ab5616c19'),
 ('selected_references','inputs/references/selected-reference-rows.jsonl','d60d48e42f29608dfd9a0a1c04bc771fb544460966f9f9008628732a9d23dc49'),
]
EXPECTED={
 'source':'6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3',
 'build':'96396de7fa14197a6759583d8b7bb053c412e08f1051b947502c1cc38b185229',
 'archive':'65703d9d857ec15d2d1e8ae93cac605ef5f46964b5a1d836f1af1b0eaf916047',
 'executable':'5dc0b921b41527b973b498d682f84b010b6f41f8acf4c11b1eec4a7305eef3b2',
 'metallib':'3996691d7674278d3cfffd4d6444b2412948824b78a28437af2da3c322a4fecd',
 'wrapper':'69dc60e47321ccca5ab09e5133d9f08620aca13a0bf15395e349e411329a0475',
 'shader':'23f6f758cda7ccc0e0651e45a298644e6b71a3c25cd5ef2f7c9cbe4ceca40628',
 'helper':'865a7399791390c3a1d27f9e8d70bb0aa9a5f8e52ebec14cd350a0c3fe258b81',
}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def verify(path,want,label):
 if not path.is_file():raise RuntimeError(f'missing {label}: {path}')
 got=sha(path)
 if got!=want:raise RuntimeError(f'{label} hash mismatch: expected {want}, got {got}')
 return got
def command(mode):
 values=[(n,h) for n,_,h in INPUTS]
 args=[str(BIN),mode]
 flags=['--capture-sha256','--package-sha256','--manifest-sha256','--receipt-sha256','--mp-reference-sha256','--mp-arrays-sha256','--mp-package-manifest-sha256','--reference-export-sha256']
 for flag,(_,digest) in zip(flags,values):args.extend([flag,digest])
 args.extend(str(PKG/rel) for _,rel,_ in INPUTS)
 args.append(str(OUT))
 return args
def verify_bindings():
 verify(SRC/'frozen-source.json',EXPECTED['source'],'source173 frozen source')
 verify(BUILD/'source-build.json',EXPECTED['build'],'source173 build provenance')
 verify(BUILD/'libnumi_matter_compiler.a',EXPECTED['archive'],'source173 compiler archive')
 verify(BIN,EXPECTED['executable'],'checker executable')
 verify(PKG/'shaders/cardboard_bilinear15_capture_diagnostic.metallib',EXPECTED['metallib'],'checker metallib')
 verify(PKG/'candidate/matter/tools/cardboard_source469_material_replay_check.mm',EXPECTED['wrapper'],'source173-bound wrapper')
 verify(PKG/'candidate/matter/tools/cardboard_bilinear15_capture_diagnostic.metal',EXPECTED['shader'],'shader source')
 verify(PKG/'candidate/matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm',EXPECTED['helper'],'helper source')
 checked=[]
 for name,rel,digest in INPUTS:checked.append({'name':name,'path':str(PKG/rel),'sha256':verify(PKG/rel,digest,name)})
 if OUT.exists():raise RuntimeError(f'output must be a fresh path: {OUT}')
 free=shutil.disk_usage(PKG).free
 if free<DISK_FLOOR:raise RuntimeError(f'disk free {free} below required {DISK_FLOOR}')
 return checked,free
def run_cpu_admission():
 argv=command('--cpu-admission-469')
 record={'argv':argv,'cwd':str(PKG),'gpu_execution':False,'mode':'cpu_admission'}
 path=PKG/'validation/guarded-cpu-admission-v2.command.json'
 stdout=PKG/'validation/guarded-cpu-admission-v2.stdout.log'
 stderr=PKG/'validation/guarded-cpu-admission-v2.stderr.log'
 if any(p.exists() for p in (path,stdout,stderr)):raise RuntimeError('CPU admission receipt paths already exist; refusing overwrite')
 path.write_text(json.dumps(record,indent=2)+'\n')
 p=subprocess.run(argv,cwd=PKG,text=True,capture_output=True)
 stdout.write_text(p.stdout);stderr.write_text(p.stderr)
 if p.returncode:raise RuntimeError(f'CPU admission failed ({p.returncode}); see {stderr}')
 return json.loads(p.stdout)
def current_worker_processes():
 p=subprocess.run(['/bin/ps','-axo','pid=,command='],text=True,capture_output=True,check=True)
 markers=('numi-matter-','sparse-scan-checker','source469-material-replay-check')
 found=[]
 for line in p.stdout.splitlines():
  bits=line.strip().split(None,1)
  if not bits or not bits[0].isdigit() or int(bits[0])==os.getpid():continue
  command=bits[1] if len(bits)>1 else ''
  if any(marker in command for marker in markers):found.append(line.strip())
 return found
def lock_is_busy():
 if not LOCK.exists():return False
 with LOCK.open('a+') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  fcntl.flock(f,fcntl.LOCK_UN)
 return False
def rss_group_bytes(pgid):
 p=subprocess.run(['/bin/ps','-axo','pid=,pgid=,rss='],text=True,capture_output=True,check=True)
 total=0
 for line in p.stdout.splitlines():
  bits=line.split()
  if len(bits)==3 and bits[1].isdigit() and int(bits[1])==pgid:total+=int(bits[2])*1024
 return total
def terminate_group(proc):
 try:os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:proc.wait(timeout=5)
 except subprocess.TimeoutExpired:
  try:os.killpg(proc.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  proc.wait()
def execute(timeout_seconds):
 checked,free=verify_bindings()
 if OUT.exists():raise RuntimeError('native output exists; O_EXCL refusal')
 LOCK.parent.mkdir(parents=True,exist_ok=True)
 lockf=LOCK.open('a+')
 try:fcntl.flock(lockf,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError(f'another native worker owns {LOCK}')
 other=current_worker_processes()
 if other or lock_is_busy():raise RuntimeError('another cardboard worker is active or owns the source173 lock: '+' | '.join(other))
 argv=command('--native-469')
 proof={'argv':argv,'source_sha256':EXPECTED['source'],'build_sha256':EXPECTED['build'],'executable_sha256':EXPECTED['executable'],'input_hashes':checked,'disk_free_before_bytes':free,'memory_limit_bytes':MEMORY_LIMIT,'disk_floor_bytes':DISK_FLOOR,'timeout_seconds':timeout_seconds,'started':True}
 receipt=PKG/'validation/native469-execution-receipt.json'
 if receipt.exists():raise RuntimeError('execution receipt exists; refusing reuse')
 (PKG/'validation/native469-execution.command.json').write_text(json.dumps(proof,indent=2)+'\n')
 proc=subprocess.Popen(argv,cwd=PKG,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
 (PKG/'runs').mkdir(exist_ok=True)
 started=time.monotonic();limit_reason=None
 while proc.poll() is None:
  rss=rss_group_bytes(proc.pid);free_now=shutil.disk_usage(PKG).free
  if rss>MEMORY_LIMIT:limit_reason=f'process_group_rss_exceeded_{MEMORY_LIMIT}';terminate_group(proc);break
  if free_now<DISK_FLOOR:limit_reason=f'disk_free_below_{DISK_FLOOR}';terminate_group(proc);break
  if time.monotonic()-started>timeout_seconds:limit_reason='timeout';terminate_group(proc);break
  time.sleep(1)
 out,err=proc.communicate()
 (PKG/'validation/native469-execution.stdout.log').write_text(out)
 (PKG/'validation/native469-execution.stderr.log').write_text(err)
 receipt_data={'schema':'numi.cardboard.native469-guarded-run.v1','execution_started':True,'exit_code':proc.returncode,'limit_reason':limit_reason,'wall_seconds':time.monotonic()-started,'output_path':str(OUT),'output_exists':OUT.exists(),'gpu_execution':True,'material_qualification':False}
 receipt.write_text(json.dumps(receipt_data,indent=2)+'\n')
 print(json.dumps(receipt_data,indent=2))
 if proc.returncode:raise SystemExit(proc.returncode)
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--execute',action='store_true',help='explicitly dispatch the diagnostic Metal replay; without this flag only CPU admission runs')
 ap.add_argument('--timeout-seconds',type=int,default=7200)
 a=ap.parse_args()
 if a.timeout_seconds<1 or a.timeout_seconds>14400:raise SystemExit('timeout must be 1..14400 seconds')
 checked,free=verify_bindings()
 admission=run_cpu_admission()
 if not a.execute:
  workers=current_worker_processes();busy=lock_is_busy()
  allowed=not workers and not busy
  print(json.dumps({'status':'input_and_cpu_admission_pass_no_native_dispatch','gpu_execution':False,'launch_allowed':allowed,'launch_block_reason':None if allowed else 'active cardboard/native worker or locked source173 worker lease','active_workers':workers,'source173_lock_busy':busy,'input_hashes':checked,'disk_free_bytes':free,'disk_free_decimal_gb':free/1e9,'required_disk_floor_bytes':DISK_FLOOR,'required_memory_limit_bytes':MEMORY_LIMIT,'memory_limit_semantics':'sampled aggregate RSS for the native process group; not an OS reservation','cpu_admission':admission,'native_argv_template':command('--native-469')},indent=2))
  return
 execute(a.timeout_seconds)
if __name__=='__main__':main()
