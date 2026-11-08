from pathlib import Path
import importlib.util,json,shutil,subprocess,sys
r=Path.cwd();out=r/'artifacts/crow-rl-object-push-20261108'
spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def record(p):return {'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':p.stat().st_mode&0o777}
retained=r/'.numi/retained-runtime/object-push-final'
for relative in ('bin/metalrobo_task_rollout','bin/metalrobo_task_train','bin/metalrobo_crow_object_push_check','bin/metalrobo_crow_object_push_compile','bin/metalrobo_metal_world_contact_probe','lib/libmetalrobo.dylib','shaders/MetalRobo.metallib','shaders/GRUMemory.metallib','shaders/RetinalStream.metallib'):
 p=r/'build'/relative;q=retained/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
old=json.loads((r/'artifacts/crow-rl-optimizer-audit-20261108/runtime-recovery.json').read_text());m={'manifests':old['manifests']+['artifacts/crow-rl-object-push-20261108/manifest.json'],'native':{},'python_sources':{},'git_inputs':{},'checkpoint':{}}
for p in retained.rglob('*'):
 if p.is_file():m['native'][str(p.relative_to(retained))]={**record(p),'archive_source':str(p.relative_to(r))}
for p in (r/'python/metalrobo').rglob('*.py'):
 name=str(p.relative_to(r));m['python_sources'][name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
for name in a.git('ls-files','include').decode().splitlines():
 if not name.endswith(('.h','.hpp')):continue
 p=r/name;m['git_inputs']['source/'+name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
p=r/'artifacts/crow-rl-object-push-20261108-identity_oracle.mm';m['git_inputs']['source/identity_oracle.mm']={**record(p),'git_blob':a.git('rev-parse','HEAD:'+str(p.relative_to(r))).decode().strip()}
for p in (r/'environments/american-crow/object-push-v1').rglob('*'):
 if p.is_file():
  name=str(p.relative_to(r));m['git_inputs']['project/'+name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
ref=r/'.numi/runs/american-crow/object-push-native-11/native-evidence'
m['native_reference']={'original_root':str(ref),'files':{str(p.relative_to(ref)):record(p)for p in ref.rglob('*')if p.is_file()}}
ref=r/'.numi/runs/american-crow/object-push-throughput-01';m['learning_reference']={'command':json.loads((ref/'plan.json').read_text())['command'],'files':{str(p.relative_to(ref)):record(p)for p in ref.rglob('*')if p.is_file()},'summary':json.loads((ref/'summary.json').read_text())}
for p in ref.rglob('*'):
 if p.is_file() and p.suffix in ('.policypack','.rolloutpack','.safetensors'):
  m['checkpoint']['reference-learning/'+str(p.relative_to(ref))]=str(p.relative_to(r))
m['recovery_scope']='All evidence Git content verified; actual recovered native contact/reward/partition/retina/future controls and fresh ordinary fixed-LR2-update replay. Matching OS/device/dependencies required. Engineering fixture only; no learned manipulation, ecological or sustained throughput qualification.'
subprocess.run([sys.executable,'tools/crow_evidence_archive.py','capture',str(out.relative_to(r))],check=True)
a.write_json(out/'runtime-recovery.json',m);shutil.copy2('/private/tmp/crow-object-push-report.md',out/'README.md')
a.write_json(out/'analysis.json',{'source_commit':a.git('rev-parse','HEAD').decode().strip(),'native_summary':json.loads((r/m['native_reference']['original_root']/'summary.json').read_text()),'learning_summary':m['learning_reference']['summary'],'performance':json.loads((ref/'metrics-validation.json').read_text()),'broader_goal_complete':False})
print(json.dumps({'archive':str(out),'raw_files':len(json.loads((out/'manifest.json').read_text())['files'])}))
