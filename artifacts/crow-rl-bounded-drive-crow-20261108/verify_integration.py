from pathlib import Path
import hashlib,json,os,shutil,subprocess,time,sys
R=Path.cwd();B=R/'build/crow-bounded-drive-integration-build-01';A=R/'artifacts/crow-rl-bounded-drive-crow-20261108';T=R/'.numi/retained-runtime/bounded-drive-twofold-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
records=[]
def execute(name,cmd,env=None,timeout=300):
 start=time.monotonic()
 with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:code=subprocess.run([str(p) for p in cmd],env=env,stdout=out,stderr=err,timeout=timeout).returncode
 rec={'name':name,'command':[str(p) for p in cmd],'exit_code':code,'seconds':time.monotonic()-start};records.append(rec);write(B/'final-integration-execution.json',records);print(json.dumps(rec),flush=True);assert code==0,rec
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
execute('final-full-release',['cmake','--build','build','-j4'],timeout=1200)
for source,target in [('build/lib/libmetalrobo.dylib','libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib','metalrobo/MetalRobo.metallib'),('build/bin/metalrobo_bounded_drive_probe','probe')]:assert sha(R/source)==sha(T/target),(source,target)
for source,target in [('build/bin/metalrobo_metal_world_contact_probe','contact-probe'),('build/bin/metalrobo_crow_object_push_check','object-push-check'),('build/shaders/RetinalStream.metallib','metalrobo/RetinalStream.metallib')]:assert not (T/target).exists();shutil.copy2(R/source,T/target)
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'));env.pop('METALROBO_BOUNDED_FAILURE_CAPTURE',None)
execute('final-default-contact',[T/'contact-probe'],env)
execute('final-default-object-push',[T/'object-push-check',B/'default-object-push'],env)
execute('final-default-crow32',[B/'fixture-base',B/'default-crow32','0','1','32'],env)
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
entries=tree();reference_path='artifacts/crow-rl-ccd-128-integration-20261108/payload/manifest.json';reference=json.loads(subprocess.check_output(['git','show','HEAD:'+reference_path]))
checks={}
with Objects() as objects:
 for name in ('trace.tsv','final-q.f32','final-v.f32','final-scene.bin','scene.bin','summary.json'):
  candidates=[f for f in reference['files'] if f['path'].endswith('/integrated32/'+name)];assert len(candidates)==1,(name,len(candidates));f=candidates[0]
  old=b''.join(content(f,entries,objects));assert hashlib.sha256(old).hexdigest()==f['sha256'];new=(B/'default-crow32'/name).read_bytes();checks[name]={'byte_exact':new==old,'sha256':hashlib.sha256(new).hexdigest(),'reference_source':f['path']};assert new==old,name
loads={}
for name in ('final-default-contact','final-default-object-push','final-default-crow32'):
 q=[s for s in (B/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s];assert q and all(s.endswith(str(T/'libmetalrobo.dylib')) for s in q);loads[name]=q
old=list(__import__('csv').DictReader((B/'native-v2.tsv').open(),delimiter='\t'));new=list(__import__('csv').DictReader((B/'twofold-constitutive.tsv').open(),delimiter='\t'));legacy_exact=[q for q in old if q['bounded']=='0']==[q for q in new if q['bounded']=='0'];assert legacy_exact
policy=R/'environments/american-crow/action-holds-v1/policies/deployment97.policypack';assert sha(policy)=='6a009417eac3e04831c95f46c24e78f943e825892124b972a58b5ee9eeed0389'
report={'source_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'commands':records,'full_release_pass':True,'full_release_runtime_equals_tested_snapshot':True,'default_native_contact_pass':True,'default_native_object_push_pass':True,'default_crow640transition_replay':checks,'legacy448constitutive_rows_exact':legacy_exact,'loader_lines':loads,'protected_policy_sha256':sha(policy),'df_before':before,'df_after':subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True),'policy_or_profile_promoted':False,'full_crow_qualified':False}
write(A/'verification.json',report);write(B/'final-integration-analysis.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('commands','loader_lines')}),flush=True)
