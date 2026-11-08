from pathlib import Path
import json,os,subprocess,time,hashlib
R=Path.cwd();A=R/'artifacts/crow-active-navigation-transfer-20261108';B=R/'build/crow-active-navigation-transfer-01'
policies={'initial':R/'build/crow-active-navigation-train-smoke-01/initial.policypack',
          'updated':R/'build/crow-active-navigation-train-smoke-01/updated.policypack',
          'transferred':B/'transferred.policypack'}
results=[]
for course,seed in [('held-out-a',20261109),('held-out-b',20261110)]:
 for which,policy in policies.items():
  cmd=[str(R/'build/bin/metalrobo_task_rollout'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15',
       '--birdflow-navigation-course',course,'--envs','8','--steps','1600','--repeats','1','--chunk','1',
       '--physics-substeps','4','--seed',str(seed),'--no-scheduled-resets','--minimum-difficulty-band','10','--maximum-difficulty-band','10',
       '--policy-pack',str(policy),'--visual-observation-config','assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json']
  name='full-'+course+'-'+which;start=time.monotonic()
  r=subprocess.run(cmd,cwd=R,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},text=True,capture_output=True,timeout=300)
  (B/(name+'.stdout')).write_text(r.stdout);(B/(name+'.stderr')).write_text(r.stderr)
  try:
   p=json.loads(r.stdout)
   result={k:v for k,v in p.items() if ('navigation' in k or k in ('minimum_sampled_difficulty_band','maximum_sampled_difficulty_band','failed_environment_steps','termination_count','environment_control_steps_per_second','steps_per_repeat','environments'))}
  except ValueError:result={}
  results.append({'name':name,'argv':cmd,'returncode':r.returncode,'seconds':time.monotonic()-start,'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),'metrics':result,'stderr':r.stderr[-2000:]})
  (A/'full-journey-result.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
  print(name,r.returncode,flush=True)
