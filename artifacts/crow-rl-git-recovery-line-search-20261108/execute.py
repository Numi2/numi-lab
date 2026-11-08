from pathlib import Path
import csv,gzip,hashlib,importlib.util,json,os,shutil,stat,struct,subprocess,sys,time
import numpy as np
R=Path.cwd();ar='artifacts/crow-rl-line-search-refinement-20261108';wr='artifacts/crow-rl-line-search-witness-20261108'
br='build/crow-line-search-refinement-build-01';wb='build/crow-line-search-witness-build-01';gr='artifacts/crow-rl-git-recovery-line-search-20261108'
A=R/ar;G=R/gr;X=R/'build/crow-line-search-recovery-01'
sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
entries=archive.tree();X.mkdir();mapped={}
with archive.Objects() as objects:
 def committed(path):return objects.read(entries[path][1])
 manifest_raw=committed(ar+'/payload/manifest.json');manifest=json.loads(manifest_raw);assert manifest_raw==(A/'payload/manifest.json').read_bytes()
 for f in manifest['files']:
  raw=b''.join(archive.content(f,entries,objects));assert (sha(raw),len(raw))==(f['sha256'],f['bytes'])
  output=any('/'+part+'/' in f['path'] for part in ['observations','integration-observations','qualification-observations'])
  for p in [X/'original'/f['path']]+([] if output else [X/f['path']]):
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(f['mode']);os.utime(p,ns=(f['mtime_ns'],f['mtime_ns']))
  mapped.setdefault((f['sha256'],f['bytes']),dict(original=f['path']))
 for root in [ar,wr]:
  for path in entries:
   if path.startswith(root+'/') and len(Path(path).parts)==3:
    p=X/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(committed(path))
 def sources(root,old_revision=None):
  plan=json.loads(committed(root+'/plan.json'));old=archive.tree(old_revision) if old_revision else entries
  for path,h in plan['sources'].items():
   tree=old if path.startswith('src/metal/') and old_revision else entries
   raw=objects.read(tree[path][1]);assert sha(raw)==h,(path,sha(raw),h)
   p=X/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 sources(wr,'c55197a2')
 aliases=json.loads(committed(ar+'/runtime-recovery.json'))['aliases']
 for target,source in aliases.items():
  p=X/target;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(X/source,p)
(X/wb/'observations').mkdir()
start_all=time.monotonic();execution=[];compared=[];physical_records=0
originals={f['path']:f for f in manifest['files']}
def run(name,cmd,runtime,shader,code,cap=None,timeout=180):
 env={**os.environ,'DYLD_LIBRARY_PATH':str(X/runtime),'DYLD_PRINT_LIBRARIES':'1','CROW_PROBE_METALLIB':str(X/shader),'PYTHONDONTWRITEBYTECODE':'1'}
 if cap:env['METALROBO_QUALITY_FAILURE_CAPTURE']=str(X/cap)
 start=time.monotonic()
 with (G/(name+'.stdout')).open('xb') as stdout,(G/(name+'.stderr')).open('xb') as stderr:
  got=subprocess.run([str(v) for v in cmd],cwd=X,env=env,stdout=stdout,stderr=stderr,timeout=timeout).returncode
 execution.append(dict(name=name,returncode=got,seconds=time.monotonic()-start));write(G/'execution.json',execution);print(json.dumps(execution[-1]),flush=True);assert got==code,(name,got,code)
 if runtime is not None:assert any(s.rstrip().endswith(str(X/runtime/'libmetalrobo.dylib')) for s in (G/(name+'.stderr')).read_text().splitlines()),name
def exact(path):
 f=originals[path];assert sha((X/path).read_bytes())==f['sha256'],path;compared.append(path)
def compare_dir(path):
 for name,f in originals.items():
  if str(Path(name).parent)==path and not name.endswith('runtime.json'):exact(name)
# One of the two identical observer-only failures is executed, using its exact old runtime.
run('observer-api',[X/wb/'reference/crow-fixture',X/wb/'observations/api-0','0','1','32','2','api-default'],
 Path(wb)/'runtime',Path(wb)/'runtime/metalrobo/MetalRobo.metallib',1,wb+'/observations/api-0-capture.bin')
compare_dir(wb+'/observations/api-0');exact(wb+'/observations/api-0-capture.bin')
with archive.Objects() as objects:
 # Switch only the restored source witnesses to the later candidate phase.
 sources(ar)
 plan=json.loads(committed(ar+'/integration-plan.json'))
 for p,h in plan['files'].items():assert sha((X/p).read_bytes())==h,p
 inputs=json.loads(committed(ar+'/inputs.json'));native_rows=json.loads(committed(ar+'/component-analysis.json'))
# Execute each unique candidate frozen operator once; duplicate baseline/repeat captures are recovered and hash checked.
for name,f in inputs.items():
 out=Path(br)/'observations'/(name+'-candidate-0');(X/out).parent.mkdir(parents=True,exist_ok=True)
 # Standalone probe uses Metal directly, so no dynamic project library is required.
 cmd=[X/br/'probe',X/br/'runtime/metalrobo/MetalRobo.metallib',X/f['path'],X/out,'0'];start=time.monotonic()
 with (G/(name+'.stdout')).open('xb') as stdout,(G/(name+'.stderr')).open('xb') as stderr:
  code=subprocess.run([str(v) for v in cmd],cwd=X,stdout=stdout,stderr=stderr,timeout=30).returncode
 execution.append(dict(name='component-'+name,returncode=code,seconds=time.monotonic()-start));write(G/'execution.json',execution);assert code==0
 for leaf in ['dispatch.bin','status.bin','velocity.f32','impulse.f32']:exact(str(out/leaf))
# Re-execute the eight regression jobs. The nondeterministic wall-time prefix is recovered, not re-run.
prior=json.loads((X/ar/'integration-execution.json').read_text());assert len(prior)==9 and prior[-1]['returncode'] is None
for job in prior[:8]:
 cmd=[str(X)+v[len(str(R)):] if v.startswith(str(R)+'/') else v for v in job['argv']]
 (X/br/'integration-observations').mkdir(exist_ok=True)
 cap=br+'/integration-observations/strict-capture.bin' if job['name']=='strict' else None
 run('regression-'+job['name'],cmd,Path(br)/'runtime',Path(br)/'runtime/metalrobo/MetalRobo.metallib',job['returncode'],cap)
compare_dir(br+'/integration-observations/default');compare_dir(br+'/integration-observations/strict');exact(br+'/integration-observations/strict-capture.bin')
for leaf in ['motor.tsv','zero.tsv']:
 path=br+'/integration-observations/'+leaf
 def records(p):return [{k:v for k,v in r.items() if k!='gpu_ms'} for r in csv.DictReader(p.open(),delimiter='\t')]
 before=records(X/'original'/path);assert before==records(X/path);physical_records+=len(before)
assert physical_records==624
# Execute one full coupled repeat; the other identical repeat is fully Git recovered and checked.
qualified=json.loads((X/ar/'qualification-execution.json').read_text());assert len(qualified)==2 and qualified[0]['returncode'] in [0,1]
(X/br/'qualification-observations').mkdir()
cmd=[X/br/'reference/crow-fixture',X/br/'qualification-observations/api-0','0','1','32','2','api-default']
run('full-coupled-api',cmd,Path(br)/'runtime',Path(br)/'runtime/metalrobo/MetalRobo.metallib',qualified[0]['returncode'],br+'/qualification-observations/api-0-capture.bin',600)
compare_dir(br+'/qualification-observations/api-0')
cap=br+'/qualification-observations/api-0-capture.bin'
if cap in originals:exact(cap)
# Fresh independent FP64 analysis owns no transition.
spec=importlib.util.spec_from_file_location('crow_recovery_ref',X/wr/'independent.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref);cal=ref.calibrate();audits=[]
for name,f in inputs.items():
 raw=(X/f['path']).read_bytes();d=struct.unpack_from('<16I8f',raw,32);nv,rs,nb=d[2:5];at=256+128*nb;blocks=raw[256:at]
 def take(n):
  global at
  value=np.frombuffer(raw,dtype='<f4',count=n,offset=at).copy();at+=4*n;return value
 M=take(nv*nv);J=take(rs*nv);bias=take(rs);free=take(nv);warm=take(nv)
 packed=struct.pack('<8I',0x53504331,1,nv,rs,nb,20,0,0)+blocks+b''.join(v.tobytes() for v in [M,J,bias,free,warm])
 c=ref.parse(packed);p=X/br/'observations'/(name+'-candidate-0');status=struct.unpack('<12I12f',(p/'status.bin').read_bytes())
 if status[0]==0:
  v=np.fromfile(p/'velocity.f32',dtype='<f4')[:nv].astype(float);imp=np.fromfile(p/'impulse.f32',dtype='<f4')[:rs].astype(float);q=ref.evaluate(c,v)
  av=c['A']@(v-c['free']);jp=c['J'].T@imp;published=float(np.max(np.abs(av-jp))/(1+np.max(np.abs(av))+np.max(np.abs(jp))))
  old=next(r for r in native_rows if r['name']==name and r['arm']=='candidate')
  assert q['res']==old['independent_projected_residual'] and published==old['independent_published_residual']
  assert q['res']<=d[16] and published<=d[16] and q['feas']<=d[17]
  audits.append(dict(name=name,projected=q['res'],published=published,gate_passed=True))
 else:
  assert name=='substep7' and status[0]==7
  assert all((p/n).read_bytes()==b'\xCD'*(4*max(nv,rs)) for n in ['velocity.f32','impulse.f32'])
  audits.append(dict(name=name,rejection_unchanged=True))
c=ref.parse((X/wb/'failed-operator.bin').read_bytes());v,q,history=ref.reference(c)
witness=json.loads((X/wr/'analysis.json').read_text());rounded=ref.evaluate(c,v.astype('<f4').astype(float))
assert history==witness['reference_residual_history'] and rounded['res']==witness['reference_rounded_residual']
write(G/'independent-audits.json',dict(calibration=cal,candidate_cases=audits,observer_reference_history_exact=True,observer_rounded_residual=rounded['res']))
# Every original, including the timed-out prefix and repeats not executed, is independently read from Git and metadata checked.
for f in manifest['files']:
 p=X/'original'/f['path'];s=p.stat();assert (sha(p.read_bytes()),s.st_size,stat.S_IMODE(s.st_mode),s.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
blobs=G/'recovery-blobs';blobs.mkdir();inventory=[]
for p in sorted(X.rglob('*')):
 if not p.is_file():continue
 assert not p.is_symlink();s=p.stat();raw=p.read_bytes();key=(sha(raw),len(raw));backing=mapped.get(key)
 if backing is None:
  dest=blobs/(key[0]+'.gz')
  if not dest.exists():dest.write_bytes(gzip.compress(raw,mtime=0))
  backing=dict(gzip=str(dest.relative_to(R)));mapped[key]=backing
 inventory.append(dict(path=str(p.relative_to(X)),sha256=key[0],bytes=key[1],mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,backing=backing))
write(G/'temporary-inventory.json',dict(schema='crow-line-search-temporary-v1',root=str(X.relative_to(R)),source_manifest=ar+'/payload/manifest.json',source_manifest_sha256=sha(manifest_raw),files=inventory))
assert len(execution)==16
result=dict(schema='crow-line-search-git-recovery-v1',manifest_sha256=sha(manifest_raw),native_commands=len(execution),physical_and_numerical_files_exact=len(compared),
 motor_physical_records_exact=physical_records,unique_candidate_cases_independently_audited=6,observer_reference_history_exact=True,
 original_files_git_recovered_and_metadata_checked=len(manifest['files']),timed_out_prefix_replayed=False,
 duplicate_full_coupled_repeats_executed=1,original_duplicate_repeats_git_verified=True,
 temporary_files=len(inventory),temporary_bytes=sum(f['bytes'] for f in inventory),seconds=time.monotonic()-start_all,
 all_passed=True,scope='Exact archived runtime execution. No fresh Git-source rebuild. One of identical repeats executed; all originals Git/hash/mode/mtime verified.',
 goal_complete=False)
write(G/'result.json',result);print(json.dumps(result),flush=True)
