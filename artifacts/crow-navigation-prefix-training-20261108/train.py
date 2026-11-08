from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path.cwd();B=R/'build/crow-navigation-prefix-training-01';A=R/'artifacts/crow-navigation-prefix-training-20261108'
policy=R/'build/crow-navigation-feedback-qualification-01/versioned.policypack';shader=R/'build/crow-navigation-direct-versioned-01/direct.metallib'
cmd=[str(R/'build/bin/metalrobo_task_train'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15',
 '--birdflow-journey-teacher','--birdflow-journey-student-authority','0','--birdflow-navigation-course','training',
 '--visual-observation-config','assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json',
 '--metallib',str(shader),'--minimum-difficulty-band','10','--maximum-difficulty-band','10',
 '--envs','16','--steps','800','--updates','8','--chunk','1','--seed','20261111',
 '--mlx-python',str(R/'python/.venv/bin/python'),'--python-root',str(R/'python'),
 '--policy-pack',str(policy),'--updated-policy-pack',str(B/'updated.policypack'),'--rollout-pack',str(B/'rollout.bin'),
 '--navigation-self-imitation-minimum-waypoint','1','--update-epochs','2','--learning-rate','0.0001','--fixed-learning-rate']
plan={'argv':cmd,'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),
      'input_policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),
      'hypothesis':'Teacher-executed first-waypoint prefixes can bootstrap autonomous gate approach.',
      'qualification':'Exploratory; never promote without independent full-route qualification.',
      'teacher_boundary':'Direct non-deployed teacher supplies flight control; head actions remain policy-controlled.'}
(A/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
start=time.monotonic()
with (B/'train.stdout').open('w') as out,(B/'train.stderr').open('w') as err:
 try:r=subprocess.run(cmd,cwd=R,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=out,stderr=err,timeout=600);code=r.returncode
 except subprocess.TimeoutExpired:code=None
(A/'execution.json').write_text(json.dumps({'returncode':code,'seconds':time.monotonic()-start},indent=2)+'\n')
print('finished',code,flush=True)
