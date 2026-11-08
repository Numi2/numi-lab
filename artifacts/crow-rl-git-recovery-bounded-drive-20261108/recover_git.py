from pathlib import Path
import gzip,hashlib,json,os,stat,subprocess,sys,time
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-bounded-drive-20261108';X=R/'build/crow-bounded-drive-git-recovery-01';Brel=Path('build/crow-bounded-drive-integration-build-01')
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((G/'plan.json').read_text());entries=tree();manifest_path=plan['manifest']
with Objects() as objects:
 raw=objects.read(entries[manifest_path][1]);assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];m=json.loads(raw)
 assert m['summary']['source_bytes']<=plan['restore_max_bytes'];X.mkdir(exist_ok=False)
 for f in m['files']:
  target=X/f['path'];target.parent.mkdir(parents=True,exist_ok=True);data=b''.join(content(f,entries,objects));assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'];target.write_bytes(data);target.chmod(f['mode']);os.utime(target,ns=(f['mtime_ns'],f['mtime_ns']))
print(json.dumps({'restored_files':len(m['files']),'restored_bytes':m['summary']['source_bytes']}),flush=True)
B=X/Brel;runtime=lambda name:X/'.numi/retained-runtime'/name
T1=runtime('bounded-drive-integration-20261108');T2=runtime('bounded-drive-integration-v2-20261108');T3=runtime('bounded-drive-witness-20261108');T4=runtime('bounded-drive-precision-20261108');T6=runtime('bounded-drive-strict-20261108');T7=runtime('bounded-drive-twofold-20261108')
commands=[];checks=[]
def execute(name,T,cmd,expected,capture=False):
 env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'))
 for key in ('METALROBO_METALLIB','METALROBO_CCD_FAILURE_CAPTURE','METALROBO_BOUNDED_FAILURE_CAPTURE'):env.pop(key,None)
 if capture:env['METALROBO_BOUNDED_FAILURE_CAPTURE']=str(X/(name+'.motorsnp'))
 start=time.monotonic()
 with (G/(name+'.stdout')).open('xb') as out,(G/(name+'.stderr')).open('xb') as err:
  try:code=subprocess.run([str(p) for p in cmd],stdout=out,stderr=err,env=env,timeout=300).returncode
  except subprocess.TimeoutExpired:code=None
 loads=[s for s in (G/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 rec={'name':name,'command':[str(p) for p in cmd],'expected_exit_code':expected,'exit_code':code,'seconds':time.monotonic()-start,'loader_lines':loads};commands.append(rec);write(G/'execution.json',commands);print(json.dumps(rec),flush=True)
 assert code==expected and loads and all(s.endswith(str(T/'libmetalrobo.dylib')) for s in loads),rec

def exact(a,b):
 check={'actual':str(a.relative_to(X)),'reference':str(b.relative_to(X)),'byte_exact':a.read_bytes()==b.read_bytes(),'sha256':sha(a)};checks.append(check);write(G/'comparisons.json',checks);assert check['byte_exact'],check

def directory_exact(actual,reference):
 for p in sorted(reference.iterdir()):
  if p.is_file() and p.name!='runtime.json':exact(actual/p.name,p)
execute('original-capacity-failure',T1,[T1/'probe',T1/'metalrobo/MetalRobo.metallib',X/'original.tsv'],1)
exact(X/'original.tsv',B/'native.tsv')
execute('v2-constitutive',T2,[T2/'probe',T2/'metalrobo/MetalRobo.metallib',X/'v2.tsv'],0)
exact(X/'v2.tsv',B/'native-v2.tsv')
execute('original-motor-failure',T3,[B/'fixture-bounded',X/'original-motor-failure','0','1','32'],1,True)
directory_exact(X/'original-motor-failure',B/'crow-witness');exact(X/'original-motor-failure.motorsnp',B/'crow-motor-failure.motorsnp')
execute('strict-quantization-failure',T6,[B/'fixture-bounded',X/'strict-quantization-failure','0','1','32'],1,True)
directory_exact(X/'strict-quantization-failure',B/'strict-crow32');exact(X/'strict-quantization-failure.motorsnp',B/'strict-crow32.motorsnp')
execute('current-arithmetic',T7,[B/'certificate-v2',T7/'metalrobo/MetalRobo.metallib',B/'precision-v2-crow32.motorsnp',X/'arithmetic.f32'],0)
exact(X/'arithmetic.f32',B/'twofold-arithmetic.f32')
execute('known-bad-rejection',T7,[T7/'probe',T4/'metalrobo/MetalRobo.metallib',X/'negative.tsv'],1,True)
assert 'Native motor reference gate failed' in (G/'known-bad-rejection.stderr').read_text()
execute('current-constitutive',T7,[T7/'probe',T7/'metalrobo/MetalRobo.metallib',X/'current.tsv'],0,True)
exact(X/'current.tsv',B/'twofold-constitutive.tsv')
execute('current-terminal',T7,[B/'fixture-terminal',X/'current-terminal','0','1','32'],1)
directory_exact(X/'current-terminal',B/'terminal-crow32')
execute('default-physical-replay',T7,[B/'fixture-base',X/'default-physical-replay','0','1','32'],0)
directory_exact(X/'default-physical-replay',B/'default-crow32')
# Every temporary byte has a committed archive equivalent or a new compressed
# recovery blob. Commit those blobs before the separate exact cleanup step.
known={(f['sha256'],f['bytes']):f['path'] for f in m['files']};inventory=[];(G/'recovery-blobs').mkdir(exist_ok=False)
for p in sorted(X.rglob('*')):
 if not p.is_file():continue
 st=p.lstat();assert stat.S_ISREG(st.st_mode);h=sha(p);item={'path':str(p.relative_to(R)),'bytes':st.st_size,'sha256':h,'mode':stat.S_IMODE(st.st_mode),'mtime_ns':st.st_mtime_ns}
 if (h,st.st_size) in known:item['archive_source']=known[(h,st.st_size)]
 else:
  dest=G/'recovery-blobs'/(h+'.gz')
  if not dest.exists():dest.write_bytes(gzip.compress(p.read_bytes(),compresslevel=6,mtime=0))
  item['recovery_blob']=str(dest.relative_to(R))
 inventory.append(item)
assert sum(f['bytes'] for f in inventory)<=plan['temporary_budget_bytes']
write(G/'temporary-inventory.json',inventory)
handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(X)],capture_output=True,text=True,timeout=60)
write(G/'temporary-open-handles.json',{'exit_code':handles.returncode,'stdout':handles.stdout,'stderr':handles.stderr})
assert handles.returncode in (0,1) and not handles.stdout and not handles.stderr
report={'manifest':manifest_path,'manifest_sha256':plan['manifest_sha256'],'restored_files':len(m['files']),'restored_bytes':m['summary']['source_bytes'],'native_commands':len(commands),'byte_exact_comparisons':len(checks),'all_compared_bytes_exact':all(c['byte_exact'] for c in checks),'known_bad_shader_rejected_before_further_conditions':True,'temporary_files':len(inventory),'temporary_bytes':sum(f['bytes'] for f in inventory),'new_recovery_blobs':len(list((G/'recovery-blobs').glob('*.gz'))),'new_recovery_compressed_bytes':sum(p.stat().st_size for p in (G/'recovery-blobs').glob('*.gz')),'open_handles':0,'policy_or_profile_promoted':False,'limits':'Scoped native recovery, not new scientific observations. Invalid old SIMD crow trajectories and the full object-push regression are Git-byte verified but not re-executed. All original source evidence is restored; exact physical/numerical/failure paths above are replayed.'}
write(G/'verification.json',report);print(json.dumps(report),flush=True)
