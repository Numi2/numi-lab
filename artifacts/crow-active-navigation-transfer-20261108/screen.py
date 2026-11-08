from pathlib import Path
import json,os,subprocess,time,hashlib
R=Path.cwd(); A=R/'artifacts/crow-active-navigation-transfer-20261108'; B=R/'build/crow-active-navigation-transfer-01'
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'};results=[]
for course,seed in [('held-out-a',20261109),('held-out-b',20261110)]:
 for which in ('transferred',):
  cmd=[str(R/'build/bin/metalrobo_task_rollout'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15',
       '--birdflow-navigation-course',course,'--envs','8','--steps','256','--repeats','1','--chunk','1',
       '--physics-substeps','4','--seed',str(seed),'--no-scheduled-resets',
       '--policy-pack',str(B/(which+'.policypack')),
       '--visual-observation-config','assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json']
  name=course+'-'+which
  started=time.monotonic();r=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=300)
  (B/(name+'.stdout')).write_text(r.stdout);(B/(name+'.stderr')).write_text(r.stderr)
  try:
   p=json.loads(r.stdout)
   result={k:v for k,v in p.items() if ('navigation' in k or k in ('failed_environment_steps','termination_count','environment_control_steps_per_second','steps_per_repeat','environments'))}
  except ValueError:result={}
  results.append({'name':name,'argv':cmd,'returncode':r.returncode,'seconds':time.monotonic()-started,'policy_sha256':hashlib.sha256((B/(which+'.policypack')).read_bytes()).hexdigest(),'metrics':result,'stderr':r.stderr[-2000:]})
  (A/'screen-result.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
  print(name,r.returncode,flush=True)
