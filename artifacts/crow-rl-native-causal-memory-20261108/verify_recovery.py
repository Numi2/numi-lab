from pathlib import Path
import hashlib,importlib.util,json,os,stat,struct,subprocess
import mlx.core as mx
import numpy as np
root=Path.cwd();base=root/'artifacts/crow-rl-native-causal-memory-20261108';tmp=Path('/private/tmp/crow-native-causal-memory-git-recovery-20261108')
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
m=json.loads((base/'runtime-recovery.json').read_text());plan={'budget':'One complete12-arm245760-transition fixed-policy native evaluation,600second bound, no retries','prediction':'Git-restored binary/library/shaders/policy/product inputs reproduce every native evidence file byte-exact, including paired outcomes and GRU intervention worlds. Adjacent restored library must actually load.','scope':'Fixed geometry and3 allocation/action seeds, matching device/OS. Full lifelong behavior remains unqualified.'}
if (base/'verification-plan.json').exists():
 if json.loads((base/'verification-plan.json').read_text())!=plan:raise RuntimeError('Verification plan differs')
else:(base/'verification-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
def sha(p):return a.file_hash(p)
def native(p):
 data=p.read_bytes();offset=104;parts=[]
 if data[:8]!=b'NUMILF01' or hashlib.sha256(data[64:]).digest()!=data[32:64]:raise RuntimeError('Invalid native life envelope')
 for _ in range(3):
  size,=struct.unpack_from('<Q',data,offset);offset+=8;b=data[offset:offset+size]
  if len(b)!=size or hashlib.sha256(b[64:]).digest()!=b[32:64]:raise RuntimeError('Invalid native component checksum')
  parts.append(hashlib.sha256(b).hexdigest());offset+=size
 if offset!=len(data):raise RuntimeError('Wrong native section lengths')
 return parts
env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')};env['DYLD_PRINT_LIBRARIES']='1'
def run(name,c,timeout,json_output=True):
 with (base/(name+'.command.json')).open('x') as f:f.write(json.dumps(c,indent=2)+'\n')
 p=subprocess.run(c,capture_output=True,text=True,env=env,timeout=timeout)
 for suffix,content in [('stdout',p.stdout),('stderr',p.stderr)]:
  with (base/(name+'.'+suffix)).open('x') as f:f.write(content)
 with (base/(name+'.receipt.json')).open('x') as f:f.write(json.dumps({'returncode':p.returncode},indent=2)+'\n')
 if p.returncode:raise RuntimeError(name+' failed')
 loaded=[line for line in p.stderr.splitlines() if line.startswith('dyld[') and 'libmetalrobo.dylib' in line]
 if not loaded or any(not line.endswith(str(tmp/'lib/libmetalrobo.dylib')) for line in loaded):raise RuntimeError(name+' wrong library loaded')
 return {'summary':json.loads(p.stdout) if json_output else p.stdout,'loaded_library':loaded,'library_override':False}
try:
 dest=tmp/'replay'
 c=[str(tmp/'bin/metalrobo_crow_memory_causal_check'),str(dest),str(tmp/'input/fixed-policy81.policypack'),str(tmp/'project/environments/american-crow/cued-foraging-v1/calibrated-ports.bin')]
 causal=run('causal-recovery',c,600,json_output=False)
 for name,record in m['causal_reference'].items():
  if sha(dest/name)!=record['sha256']:raise RuntimeError('Recovered native causal evidence differs: '+name)
 actual={str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}
 if actual!=set(m['causal_reference']):raise RuntimeError('Recovered native evidence file set differs')
 files=[];dirs=[]
 for directory,subdirs,names in os.walk(tmp,followlinks=False):
  for name in subdirs:
   p=Path(directory)/name
   if p.is_symlink():raise RuntimeError('Unexpected recovery symlink')
   dirs.append(p)
  for name in names:
   p=Path(directory)/name;s=p.lstat()
   if not stat.S_ISREG(s.st_mode):raise RuntimeError('Unexpected recovery file type')
   files.append({'path':str(p),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'sha256':sha(p)})
 handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
 if handles.returncode not in (0,1) or handles.stdout or handles.stderr:raise RuntimeError('Open recovery handles')
 for r in files:
  p=Path(r['path']);s=p.lstat()
  if s.st_size!=r['bytes'] or stat.S_IMODE(s.st_mode)!=r['mode'] or s.st_mtime_ns!=r['mtime_ns'] or sha(p)!=r['sha256']:raise RuntimeError('Recovery changed during cleanup')
 report={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'plan':plan,'causal':causal,'native_reference_files_exact':len(m['causal_reference']),'restoration':json.loads((tmp/'runtime-restoration.json').read_text()),'temporary_cleanup':{'status':'validated','files':files,'open_handles':0,'df_before':subprocess.check_output(['df','-k','.'],text=True)}}
 (base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 for r in files:Path(r['path']).unlink()
 for p in sorted(dirs,key=lambda p:len(p.parts),reverse=True):p.rmdir()
 tmp.rmdir();report['temporary_cleanup'].update(status='removed',df_after=subprocess.check_output(['df','-k','.'],text=True));(base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':'passed','native_reference_files_exact':len(m['causal_reference']),'temporary_files_removed':len(files),'temporary_bytes_removed':sum(r['bytes'] for r in files)}))
except Exception as error:
 (base/'verification-failure.json').write_text(json.dumps({'error':str(error),'type':type(error).__name__,'plan':plan},indent=2)+'\n');raise
