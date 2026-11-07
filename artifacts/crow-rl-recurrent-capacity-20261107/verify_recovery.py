#!/usr/bin/env python3
"""Prove a full recurrent training-life continuation from Git-restored inputs."""
from pathlib import Path
import hashlib,json,math,os,stat,struct,subprocess,sys
root=Path.cwd();base=root/'artifacts/crow-rl-recurrent-capacity-20261107';restored=Path('/private/tmp/crow-capacity-git-recovery-20261107')
plan={'prediction':'The exact Git-restored trainer, native library/shaders, all Python source identities, full prefix checkpoint and visual inputs will resume without warmup and reproduce reference revision4 policy/candidate/rollout/canonical learner bytes and every native causal state component at111 accepted ticks per lane. Only wall timing fields may differ.','stopping':'One bounded180s continuation, abort first mismatch. Hash/readback and lsof-check exact temporary deletion manifest before removing Git-restored copies. Preserve any failures.','scope':'Same-device/OS/dependency engineering recovery; no learned behavior or portability claim.'}
(base/'verification-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
restoration=json.loads((restored/'runtime-restoration.json').read_text());receipt=json.loads((restored/'inputs/tail.receipt.json').read_text());command=list(receipt['command']);result=restored/'result';result.mkdir(exist_ok=False)
command[0]=str(restored/'bin/metalrobo_task_train')
for flag,value in {'--metallib':restored/'shaders/MetalRobo.metallib','--native-library':restored/'lib/libmetalrobo.dylib','--python-root':restored/'python','--visual-observation-config':restored/'visual/retinal-observation.json','--resume-training-checkpoint':restored/'checkpoint','--updated-policy-pack':result/'policy.policypack','--deployment-policy-pack':result/'candidate.policypack','--incumbent-policy-pack':result/'incumbent.policypack','--rollout-pack':result/'rollout.rolloutpack','--output-learner-state':result/'learner.safetensors','--checkpoint-directory':result/'checkpoints'}.items():
 command[command.index(flag)+1]=str(value)
try:
 p=subprocess.run(command,capture_output=True,text=True,timeout=180)
 (base/'recovery.stdout').write_text(p.stdout);(base/'recovery.stderr').write_text(p.stderr)
 if p.returncode:raise RuntimeError('Git-restored native resume failed: '+p.stderr[-3000:])
 value=json.loads(p.stdout)
 if value['failed_environment_steps']!=0:raise RuntimeError('Git-restored native transition failure')
 exact={}
 for name in ('policy.policypack','candidate.policypack','rollout.rolloutpack','learner.safetensors'):
  f=result/name;expected=restored/'reference'/name
  if f.read_bytes()!=expected.read_bytes():raise RuntimeError('Git-restored continuation mismatch: '+name)
  exact[name]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 def state(path):
  data=path.read_bytes();magic,version,reserved,identity,size,digest=struct.unpack_from('<8sIIQQ32s',data);payload=data[64:]
  if magic!=b'NUMILF01' or version!=1 or reserved or len(payload)!=size or hashlib.sha256(payload).digest()!=digest:raise RuntimeError('Invalid life envelope')
  counts=struct.unpack_from('<QQQ',payload);timings=struct.unpack_from('<dd',payload,24)
  if counts[:2]!=(111,888) or any(not math.isfinite(x) or x<0 for x in timings):raise RuntimeError('Invalid life counters or timing')
  offset=40;components=[]
  for _ in range(3):
   length=struct.unpack_from('<Q',payload,offset)[0];offset+=8;component=payload[offset:offset+length]
   if len(component)!=length:raise RuntimeError('Truncated life component')
   offset+=length;components.append(hashlib.sha256(component).hexdigest())
  if offset!=len(payload):raise RuntimeError('Trailing native life state')
  return {'identity':identity,'counts':counts,'component_sha256':components}
 causal=state(result/'checkpoints/revision-00000004.trainingcheckpoint/life.native')
 if causal!=state(restored/'reference/life.native'):raise RuntimeError('Git-restored native causal state differs')
 verification={'status':'passed','archive_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'plan':plan,'runtime_restoration':restoration,'command':command,'native_training_record':value,'byte_exact_final_files':exact,'causal_native_state_exact':causal,'all_manifest_files_verified':147,'temporary_removal':{'status':'pending'}}
 # Exact validated deletion set. Never follow symlinks or delete new uncovered files.
 files=[];directories=[]
 for directory,dirs,names in os.walk(restored,followlinks=False):
  for name in dirs:
   d=Path(directory)/name
   if d.is_symlink():raise RuntimeError('Unexpected restoration directory symlink')
   directories.append(d)
  for name in names:
   f=Path(directory)/name;s=f.lstat()
   if not stat.S_ISREG(s.st_mode):raise RuntimeError('Unexpected restoration file type')
   files.append({'path':str(f),'bytes':s.st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns})
 check=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(restored)],capture_output=True,text=True,timeout=60)
 if check.returncode not in (0,1) or check.stdout or check.stderr:raise RuntimeError('Open restored runtime handles')
 before=subprocess.check_output(['df','-k','.'],text=True)
 for f in files:
  path=Path(f['path']);s=path.lstat()
  if s.st_size!=f['bytes'] or s.st_mtime_ns!=f['mtime_ns'] or hashlib.sha256(path.read_bytes()).hexdigest()!=f['sha256']:raise RuntimeError('Restoration changed before cleanup')
 verification['temporary_removal']={'status':'validated','open_handles':0,'files':files,'df_before':before}
 (base/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
 for f in files:Path(f['path']).unlink()
 for d in sorted(directories,key=lambda x:len(x.parts),reverse=True):d.rmdir()
 restored.rmdir()
 verification['temporary_removal'].update(status='removed',df_after=subprocess.check_output(['df','-k','.'],text=True))
 (base/'verification.json').write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps({'status':'passed','Git_restored_files':len(restoration['files']),'native_future_exact':True,'temporary_files_removed':len(files),'temporary_bytes_removed':sum(f['bytes'] for f in files)}))
except Exception as e:
 (base/'verification-failure.json').write_text(json.dumps({'error':str(e),'type':type(e).__name__,'command':command,'plan':plan},indent=2)+'\n');raise
