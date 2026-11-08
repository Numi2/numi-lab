from pathlib import Path
import json,os,subprocess,time,hashlib
R=Path.cwd(); A=R/'artifacts/crow-navigation-prefix-unclipped-20261108'
policies=[('unclipped',R/'build/crow-navigation-prefix-unclipped-01/updated.policypack.deployment.policypack')]
results=[]
for name,policy in policies:
 for course,seed in [('held-out-a',20261109),('held-out-b',20261110)]:
  cmd=[str(R/'build/bin/metalrobo_task_rollout'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15',
       '--birdflow-navigation-course',course,'--envs','8','--steps','1600','--repeats','1','--chunk','1',
       '--physics-substeps','4','--seed',str(seed),'--no-scheduled-resets','--minimum-difficulty-band','10','--maximum-difficulty-band','10',
       '--policy-pack',str(policy),'--visual-observation-config','assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json']
  label=name+'-'+course;t=time.monotonic()
  p=subprocess.run(cmd,cwd=R,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},text=True,capture_output=True,timeout=300)
  (A/(label+'.stdout')).write_text(p.stdout);(A/(label+'.stderr')).write_text(p.stderr)
  try:
   data=json.loads(p.stdout); metrics={k:v for k,v in data.items() if ('navigation' in k or k in ('failed_environment_steps','termination_count','environment_control_steps_per_second','steps_per_repeat','environments'))}
  except ValueError:metrics={}
  results.append({'name':label,'argv':cmd,'returncode':p.returncode,'seconds':time.monotonic()-t,'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),'metrics':metrics,'stderr':p.stderr[-2000:]})
  (A/'result.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
  print(label,p.returncode,flush=True)
