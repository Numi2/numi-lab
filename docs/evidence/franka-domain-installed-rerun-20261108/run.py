import datetime,hashlib,json,os,pathlib,shutil,signal,subprocess
base=pathlib.Path('/Users/numi/numi-franka-domain-smoke-20261108')
plan=json.loads((base/'plan.json').read_text()); result_path=base/'result.json'
assert not result_path.exists()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(data): return hashlib.sha256(data).hexdigest()
def hashes(): return {p:sha(pathlib.Path(p).read_bytes()) for p in plan['sha256']}
assert hashes()==plan['sha256'], 'pinned inputs changed'
env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NUMANX_'))}; env.update(plan['environment'])
r={'command':plan['argv'],'started_utc':now(),'inputs_sha256':plan['sha256']}
with (base/'train-stdout.txt').open('w') as out,(base/'train-stderr.txt').open('w') as err:
 p=subprocess.Popen(plan['argv'],cwd=plan['workspace'],env=env,stdout=out,stderr=err,start_new_session=True); r['pid']=p.pid
 try: p.wait(timeout=plan['timeout'])
 except subprocess.TimeoutExpired:
  r['timeout']=True; os.killpg(p.pid,signal.SIGTERM)
  try: p.wait(timeout=5)
  except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
r.update(ended_utc=now(),returncode=p.returncode,sha256_after=hashes())
r['identities_unchanged']=r['sha256_after']==plan['sha256']
run=pathlib.Path(plan['environment']['NUMI_RUN_DIR']); r['lock_exists_after']=(run/'.train.lock').exists()
r['success']=p.returncode==0 and not r.get('timeout') and r['identities_unchanged'] and not r['lock_exists_after']
def decision(path):
 c=subprocess.run([str(base/'decision-check'),str(path)],text=True,capture_output=True,check=True,timeout=10)
 return json.loads(c.stdout)
try:
 selection=json.loads((run/'selection/selection.json').read_text()); label=selection['selected'] if selection['selected']=='incumbent' else selection['selected_candidate_label']
 ev=run/f'selection/{label}.evidence.json'; meta=run/f'selection/{label}.evidence.json.meta.json'; pack=run/'deployment.policypack'
 evidence=ev.read_bytes(); metadata=json.loads(meta.read_text()); policy=pack.read_bytes()
 assert metadata['policy_sha256']==sha(policy) and metadata['evidence']['sha256']==sha(evidence)
 r['selected']=selection['selected']; r['real_evidence_bound']=True; r['real_decision']=decision(run)
 data=json.loads(evidence)
 assert data.get('world_source') == 'world_pack', 'expected authored non-G1 WorldPack'
 r['world_source']=data['world_source']
 # The guided balance screen is humanoid-specific. Missing fields must remain unknown.
 assert r['real_decision']['balance']=='guided held-out balance status unknown' and not r['real_decision']['should_load']
 # Copy actual bound evidence; altered copies must remain unknown and unloadable.
 fixture=base/'copied-identity-mutations'; (fixture/'selection').mkdir(parents=True)
 shutil.copy2(run/'selection/selection.json',fixture/'selection/selection.json'); shutil.copy2(pack,fixture/'deployment.policypack')
 fe=fixture/f'selection/{label}.evidence.json'; fm=fixture/f'selection/{label}.evidence.json.meta.json'
 fe.write_bytes(evidence); fm.write_text(json.dumps(metadata)+'\n')
 r['copied_matching']=decision(fixture); assert r['copied_matching']['balance']=='guided held-out balance status unknown' and not r['copied_matching']['should_load']
 fp=fixture/'deployment.policypack'; fp.write_bytes(policy+b'changed'); r['copied_changed_policy']=decision(fixture); assert r['copied_changed_policy']['balance']=='guided held-out balance status unknown' and not r['copied_changed_policy']['should_load']; fp.write_bytes(policy)
 fe.write_bytes(evidence+b' '); r['copied_changed_evidence']=decision(fixture); assert r['copied_changed_evidence']['balance']=='guided held-out balance status unknown' and not r['copied_changed_evidence']['should_load']; fe.write_bytes(evidence)
 metadata.pop('policy_sha256'); fm.write_text(json.dumps(metadata)+'\n'); r['copied_missing_binding']=decision(fixture); assert r['copied_missing_binding']['balance']=='guided held-out balance status unknown' and not r['copied_missing_binding']['should_load']
 r['decision_checks_passed']=True
except Exception as error:
 r['verification_error']=repr(error); r['success']=False
result_path.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:r.get(k) for k in ('success','returncode','started_utc','ended_utc','identities_unchanged','lock_exists_after','selected','real_decision','decision_checks_passed','verification_error')},indent=2),flush=True)
raise SystemExit(0 if r['success'] else 1)
