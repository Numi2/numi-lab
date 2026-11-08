from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-flight-source-diagnostics-20261108';B=R/'build/crow-flight-source-diagnostics-01'
policy=R/'build/crow-navigation-feedback-qualification-01/versioned.policypack'
shader=R/'build/crow-navigation-direct-versioned-01/direct.metallib'
rows=[]
for course,seed,lane in [('held-out-a',20261109,0),('held-out-a',20261109,2),('held-out-b',20261110,0),('held-out-b',20261110,5)]:
 name=f'{course}-lane-{lane}'
 cmd=[str(R/'build/bin/metalrobo_task_rollout'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15',
 '--birdflow-journey-teacher','--birdflow-journey-student-authority','0','--metallib',str(shader),
 '--birdflow-navigation-course',course,'--envs','8','--steps','1600','--repeats','1','--chunk','1',
 '--physics-substeps','4','--seed',str(seed),'--no-scheduled-resets','--minimum-difficulty-band','10','--maximum-difficulty-band','10',
 '--policy-pack',str(policy),'--visual-observation-config','assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json',
 '--state-trace',str(B/(name+'.tsv')),'--state-trace-environment',str(lane)]
 t=time.monotonic();p=subprocess.run(cmd,cwd=R,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},text=True,capture_output=True,timeout=180)
 (A/(name+'.stdout')).write_text(p.stdout);(A/(name+'.stderr')).write_text(p.stderr)
 try:
  x=json.loads(p.stdout)['first_episode_navigation']
  metric={k:x[k][lane] for k in ('maximum_waypoints_by_environment','physical_failure_by_environment','termination_reason_by_environment','steps_by_environment')}
 except (ValueError,KeyError): metric={}
 rows.append({'name':name,'argv':cmd,'returncode':p.returncode,'seconds':time.monotonic()-t,'first_episode':metric,'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest()})
 (A/'runs.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,p.returncode,metric,flush=True)
 if p.returncode:raise RuntimeError(p.stderr)
