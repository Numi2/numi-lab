from pathlib import Path
import json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-active-navigation-source-comparison-20261108';B=R/'build/crow-active-navigation-transfer-01';results=[]
for course,seed in [('held-out-a',20261109),('held-out-b',20261110)]:
 cmd=[str(R/'build/bin/metalrobo_task_rollout'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v10',
 '--birdflow-navigation-course',course,'--envs','8','--steps','1600','--repeats','1','--chunk','1','--physics-substeps','4',
 '--seed',str(seed),'--no-scheduled-resets','--minimum-difficulty-band','10','--maximum-difficulty-band','10',
 '--policy-pack',str(B/'source.policypack'),'--visual-observation-config','assets/crow_navigation_course/crow-navigation.sensor-fast.visual-observation.json']
 start=time.monotonic();p=subprocess.run(cmd,cwd=R,text=True,capture_output=True,timeout=300,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 (A/(course+'.stdout')).write_text(p.stdout);(A/(course+'.stderr')).write_text(p.stderr)
 try:
  obj=json.loads(p.stdout); metrics={k:v for k,v in obj.items() if ('navigation' in k or k in ('failed_environment_steps','termination_count','maximum_root_height','environments','steps_per_repeat'))}
 except ValueError:metrics={}
 results.append({'course':course,'seed':seed,'argv':cmd,'returncode':p.returncode,'seconds':time.monotonic()-start,'metrics':metrics})
 (A/'result.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n');print(course,p.returncode,flush=True)
