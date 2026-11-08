from pathlib import Path
import json,math,os,re,subprocess,time
R=Path.cwd(); A=R/'artifacts/crow-active-head-navigation-vision-20261108'
B=R/'build/crow-active-head-vision-controls-01';B.mkdir(exist_ok=False)
env={**os.environ,'PYTHONPATH':str(R/'python'),'PYTHONDONTWRITEBYTECODE':'1'}
old=R/'build/crow-active-head-navigation-admission-02/zero.policypack'
records=[];results={}
def run(name,cmd):
 start=time.monotonic()
 with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
  try:c=subprocess.run([str(x) for x in cmd],cwd=R,env=env,stdout=out,stderr=err,timeout=180).returncode
  except subprocess.TimeoutExpired:c=None
 records.append(dict(name=name,argv=[str(x) for x in cmd],returncode=c,seconds=time.monotonic()-start))
 (A/'controls-execution.json').write_text(json.dumps(records,indent=2,sort_keys=True)+'\n');print(name,c,flush=True)
 return c
for name in ('no-robot','flat-ground','no-robot-flat-ground'):
 config=A/'fixtures'/(name+'.json'); case=B/name;case.mkdir()
 def command(policy):
  return [R/'build/bin/metalrobo_task_rollout','--birdflow-american-crow-journey','--birdflow-journey-variant','v15',
   '--envs','1','--steps','4','--repeats','1','--chunk','1','--physics-substeps','32','--seed','20261108',
   '--no-scheduled-resets','--policy-pack',policy,'--visual-observation-config',config,
   '--crow-replay-pack',case/'replay.json','--capture-dir',case/'capture',
   '--capture-width','160','--capture-height','90','--capture-stride','4','--capture-policy-camera']
 code=run(name+'-baseline-contract',command(old))
 if code!=0:
  err=(B/(name+'-baseline-contract.stderr')).read_text()
  m=re.search(r'task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',err)
  if m:
   world,task,observation,action=m.groups();policy=case/'zero.policypack'
   init=[R/'python/.venv/bin/python','-m','metalrobo.mlx_policy_worker','initialize','--actor-observations','816',
    '--critic-observations','106','--actions','18','--world-fingerprint',world,'--task-fingerprint',task,
    '--observation-fingerprint',observation,'--action-fingerprint',action,'--policy-id','crow_v15_camera_'+name,
    '--output',policy,'--native-library',R/'build/lib/libmetalrobo.dylib','--zero-actor-output','--deterministic','--seed','20261108']
   if run(name+'-initialize',init)==0:code=run(name+'-actual',command(policy))
 p=case/'replay.json';frames=json.loads(p.read_text())['payload']['frames'] if p.exists() else []
 depth=[f['actor_observation'][-600:-456] for f in frames]
 results[name]=dict(returncode=code,frames=len(frames),zero_actions=bool(frames) and all(all(x==0 for x in f['accepted_actions']) for f in frames),
  finite=bool(depth) and all(all(math.isfinite(x) for x in d) for d in depth),
  near_counts=[sum(x<.1 for x in d) for d in depth],far_counts=[sum(x>=.999 for x in d) for d in depth],
  depth_mean=[sum(d)/len(d) for d in depth])
 (A/'controls-result.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
print(json.dumps(results,sort_keys=True),flush=True)
