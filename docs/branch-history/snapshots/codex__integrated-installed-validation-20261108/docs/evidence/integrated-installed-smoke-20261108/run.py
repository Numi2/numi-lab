import datetime,hashlib,json,os,pathlib,signal,subprocess,sys
base=pathlib.Path('/Users/numi/numi-integrated-installed-smoke-20261108')
plan=json.loads((base/'plan.json').read_text()); mode=sys.argv[1]
assert mode in ('window','train')
output=base/(mode+'-result.json'); assert not output.exists()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def hashes(): return {p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in plan['sha256']}
assert hashes()==plan['sha256'], 'pinned inputs changed'
env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NUMANX_'))}
env.update(plan['environment'])
if mode=='window': env.pop('NUMI_RUN_DIR')
record={'mode':mode,'command':plan[mode]['argv'],'started_utc':now(),'inputs_sha256':plan['sha256']}
with (base/(mode+'-stdout.txt')).open('w') as out,(base/(mode+'-stderr.txt')).open('w') as err:
 child=subprocess.Popen(plan[mode]['argv'],cwd=plan['workspace'],env=env,stdout=out,stderr=err,start_new_session=True)
 record['pid']=child.pid
 try: child.wait(timeout=plan[mode]['timeout'])
 except subprocess.TimeoutExpired:
  record['timeout']=True; os.killpg(child.pid,signal.SIGTERM)
  try: child.wait(timeout=5)
  except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
record.update(ended_utc=now(),returncode=child.returncode,sha256_after=hashes())
record['identities_unchanged']=record['sha256_after']==plan['sha256']
record['success']=child.returncode==0 and not record.get('timeout') and record['identities_unchanged']
if mode=='train':
 run=pathlib.Path(plan['environment']['NUMI_RUN_DIR'])
 record['lock_exists_after']=(run/'.train.lock').exists()
 record['success'] &= not record['lock_exists_after']
 try:
  select=json.loads((run/'selection/selection.json').read_text())
  record['selection']=select
  for label in ('incumbent','candidate'):
   evidence=(run/f'selection/{label}.evidence.json').read_bytes()
   meta=json.loads((run/f'selection/{label}.evidence.json.meta.json').read_text())
   pack=(run/f'{label}.deployment.policypack').read_bytes()
   assert meta['policy_sha256']==hashlib.sha256(pack).hexdigest()
   assert meta['evidence']['sha256']==hashlib.sha256(evidence).hexdigest()
  record['policy_evidence_hashes_valid']=True
  check=subprocess.run([str(base/'catalog-check')],cwd=plan['workspace'],env=env,text=True,capture_output=True,timeout=15)
  (base/'catalog-stdout.txt').write_text(check.stdout); (base/'catalog-stderr.txt').write_text(check.stderr)
  record['catalog_exit']=check.returncode; record['catalog']=json.loads(check.stdout)
  rows={pathlib.Path(x['path']).name:x for x in record['catalog']}
  for filename,label in (("deployment.policypack", select["selected"]), ("candidate.deployment.policypack", "candidate")):
   evidence=json.loads((run/f'selection/{label}.evidence.json').read_text())
   passed=(evidence['failed_environment_steps']==0 and evidence['height_or_tilt_termination_count']==0 and evidence['mean_tilt']<=0.35 and evidence['clean_horizon_environment_rate']>=0.5 and min(evidence['minimum_root_height_by_environment'])>=0.55)
   expected='guided held-out balance screen '+('passed' if passed else 'failed')
   assert rows[filename]['balance']==expected
   expected_role=('selected '+label) if filename=='deployment.policypack' else 'candidate experiment'
   assert rows[filename]['role']==expected_role
  record['catalog_matches_evidence']=True
  record['success'] &= check.returncode==0
 except Exception as error:
  record['verification_error']=repr(error); record['success']=False
output.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:record.get(k) for k in ('mode','success','returncode','started_utc','ended_utc','identities_unchanged','lock_exists_after','policy_evidence_hashes_valid','catalog_exit','verification_error')},indent=2),flush=True)
raise SystemExit(0 if record['success'] else 1)
