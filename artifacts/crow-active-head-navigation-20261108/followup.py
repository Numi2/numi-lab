from pathlib import Path
import hashlib,json,math,os,re,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-active-head-navigation-20261108';B=R/'build/crow-active-head-navigation-admission-02'
B.mkdir(exist_ok=False)
first=json.loads((A/'result.json').read_text())
match=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',first['policy_error'])
assert match, 'Missing runtime-owned contract'
world,task,observation,action=match.groups()
env={**os.environ,'PYTHONPATH':str(R/'python'),'PYTHONDONTWRITEBYTECODE':'1'}
policy=B/'zero.policypack';replay=B/'replay.json'
cmd1=[str(R/'python/.venv/bin/python'),'-m','metalrobo.mlx_policy_worker','initialize',
 '--actor-observations','816','--critic-observations','106','--actions','18',
 '--world-fingerprint',world,'--task-fingerprint',task,'--observation-fingerprint',observation,
 '--action-fingerprint',action,'--policy-id','crow_v15_zero_admission','--output',str(policy),
 '--native-library',str(R/'build/lib/libmetalrobo.dylib'),'--zero-actor-output','--deterministic','--seed','20261108']
cmd2=[str(R/'build/bin/metalrobo_task_rollout'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15',
 '--envs','1','--steps','8','--repeats','1','--chunk','1','--physics-substeps','32','--seed','20261108',
 '--no-scheduled-resets','--policy-pack',str(policy),'--visual-observation-config',
 'assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json','--crow-replay-pack',str(replay)]
records=[]
for name,cmd in [('initialize',cmd1),('rollout',cmd2)]:
 start=time.monotonic()
 with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
  try:code=subprocess.run(cmd,cwd=R,env=env,stdout=out,stderr=err,timeout=180).returncode
  except subprocess.TimeoutExpired:code=None
 records.append(dict(name=name,argv=cmd,returncode=code,seconds=time.monotonic()-start))
 (A/'followup-execution.json').write_text(json.dumps(records,indent=2,sort_keys=True)+'\n')
 print(json.dumps(records[-1]),flush=True)
 if code!=0: break
report=None
try:report=json.loads((B/'rollout.stdout').read_text())
except (OSError,ValueError,IndexError):pass
payload=json.loads(replay.read_text())['payload'] if replay.exists() else None
frames=payload['frames'] if payload else []
finite=bool(frames) and all(len(f['actor_observation'])==816 and all(math.isfinite(v) for v in f['actor_observation']) for f in frames)
zero=bool(frames) and all(len(f['accepted_actions'])==18 and all(v==0 for v in f['accepted_actions']) for f in frames)
checks={
 'legacy_torso_camera_rejected':first['checks']['legacy_torso_camera_rejected'],
 'feeding_policy_incompatible_contract_rejected':'incompatible_contract' in first['policy_error'] and first['checks']['feeding_policy_rejected']==False,
 'fresh_task_contract_policy_compiled':bool(records and records[0]['returncode']==0),
 'eight_transitions_and_visual_runtime':bool(report and len(records)==2 and records[1]['returncode']==0 and report['action_count']==18 and report['failed_environment_steps']==0 and report['visual_observation'] and payload and payload['frame_count']==8),
 'accepted_actions_zero':zero,'actor_observations_finite':finite,
}
result=dict(schema='crow-active-head-navigation-followup-v1',checks=checks,all_checks_passed=all(checks.values()),
 report=report,frames=len(frames),replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest() if payload else None,
 policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest() if policy.exists() else None,
 scope='Eight deterministic zero-action transitions; no learned navigation or held-out course qualification.')
(A/'followup-result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(checks,sort_keys=True),flush=True)
