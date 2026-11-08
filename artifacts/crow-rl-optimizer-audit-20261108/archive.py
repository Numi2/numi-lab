from pathlib import Path
import importlib.util,json,shutil,subprocess,sys
r=Path.cwd();out=r/'artifacts/crow-rl-optimizer-audit-20261108';trial=r/'.numi/runs/american-crow/optimizer-audit-01'
spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def record(p):return {'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':p.stat().st_mode&0o777}
retained=r/'.numi/retained-runtime/optimizer-audit-01'
for relative in ('bin/metalrobo_task_rollout','bin/metalrobo_task_train','bin/metalrobo_crow_memory_causal_check','lib/libmetalrobo.dylib','shaders/MetalRobo.metallib','shaders/GRUMemory.metallib','shaders/RetinalStream.metallib'):
 p=r/'build'/relative;q=retained/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
shutil.copy2('/private/tmp/crow_object_probe',retained/'bin/crow_object_probe_original')
old=json.loads((r/'artifacts/crow-rl-deployment-mean-20261108/runtime-recovery.json').read_text());m={'manifests':old['manifests']+['artifacts/crow-rl-optimizer-audit-20261108/manifest.json'],'native':{},'python_sources':{},'git_inputs':{},'checkpoint':{}}
for p in retained.rglob('*'):
 if p.is_file():m['native'][str(p.relative_to(retained))]={**record(p),'archive_source':str(p.relative_to(r))}
for p in (r/'python/metalrobo').rglob('*.py'):
 name=str(p.relative_to(r));m['python_sources'][name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
for p in (r/'environments/american-crow').rglob('*'):
 if p.is_file():
  name=str(p.relative_to(r));m['git_inputs']['project/'+name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
for name in a.git('ls-files','include').decode().splitlines()+['apps/CrowCameraFixture.hpp']:
 if not name.endswith(('.h','.hpp')):continue
 p=r/name;m['git_inputs']['source/'+name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
for name in ('audit.py','clipping-counterfactual.py','return-identity.py'):
 m['checkpoint']['source/'+name]=str((trial/name).relative_to(r))
for case in ('fixed65','adaptive81','precision97'):
 for p in (trial/case).iterdir():
  if p.name in ('before.policypack','before.safetensors','rollout.rolloutpack','after.policypack','after.safetensors','command.json'):
   m['checkpoint'][str(p.relative_to(r))]=str(p.relative_to(r))
m['checkpoint']['source/object-probe.mm']='.numi/runs/american-crow/object-contact-03/probe-source.mm'
p=r/'artifacts/crow-rl-optimizer-audit-20261108-identity_oracle.mm';m['git_inputs']['source/identity_oracle.mm']={**record(p),'git_blob':a.git('hash-object','-w',str(p)).decode().strip()}
m['seed_base']=2651213602;m['evaluations']={}
for case in ('fixed65','adaptive81','precision97'):
 ref=trial/'heldout'/case/'native-evidence';m['evaluations'][case]={'original_root':str(ref),'policy':str((trial/case/'before.policypack').relative_to(r)),'files':{str(p.relative_to(ref)):record(p)for p in ref.rglob('*')if p.is_file()}}
ref=r/'.numi/runs/american-crow/object-contact-03/native';m['unfinished_object_probe']={'source':'source/object-probe.mm','files':{str(p.relative_to(ref)):record(p)for p in ref.rglob('*')if p.is_file()},'scope':'Unfinished isolated mechanism probe, no reusable manipulation task or learned capability claimed.'}
m['audit_reference']={case:{'attribution':json.loads((trial/case/'attribution.json').read_text()),'update':json.loads((trial/case/'update-audit.json').read_text())}for case in ('fixed65','adaptive81','precision97')}
m['clipping_reference']=json.loads((trial/'clipping-counterfactual.json').read_text());m['returns_reference']=json.loads((trial/'return-identity.json').read_text())
m['recovery_scope']='Actual offline exact model/Adam update replays and clipping/return diagnosis from restored sources/data; complete matched3-policy native causal evaluations; source-recompiled unfinished object probe. Matching OS/device/dependencies required; no new training trial.'
subprocess.run([sys.executable,'tools/crow_evidence_archive.py','capture',str(out.relative_to(r))],check=True)
a.write_json(out/'runtime-recovery.json',m);shutil.copy2('/private/tmp/crow-optimizer-report.md',out/'README.md');shutil.copy2(trial/'heldout-analysis.json',out/'heldout-analysis.json')
a.write_json(out/'analysis.json',{'source_commit':a.git('rev-parse','HEAD').decode().strip(),'status':'diagnosed','actual_gamma':.999,'actual_lambda':1.,'exact_model_Adam_replays':3,'tensors_per_replay':53,'matched_unused_seed_evaluation':json.loads((trial/'heldout-analysis.json').read_text()),'clipping_counterfactual':m['clipping_reference'],'return_identity':m['returns_reference'],'adaptive_rate_success_qualified':False,'broader_goal_complete':False})
print(json.dumps({'archive':str(out),'raw_files':len(json.loads((out/'manifest.json').read_text())['files'])}))
