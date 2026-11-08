from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-navigation-flight-calibration-20261108';B=R/'build/crow-navigation-flight-calibration-01'
metal=R/'src/metal/LocomotionTask.metal';original=metal.read_text();needle='0.120f * forwardSpeedError +'
assert original.count(needle)==1
policy=R/'build/crow-active-navigation-transfer-01/transferred.policypack'
results=[]
try:
 for scale in ('0.120','0.350','0.700'):
  name='speed-'+scale.replace('.','-')
  metal.write_text(original.replace(needle,scale+'f * forwardSpeedError +'))
  patch=subprocess.run(['git','diff','--','src/c_api.cpp','src/metal/LocomotionTask.metal'],cwd=R,text=True,capture_output=True,check=True).stdout
  (A/(name+'.patch')).write_text(patch)
  start=time.monotonic()
  build=subprocess.run(['cmake','--build','build','--target','metalrobo_shaders','-j','2'],cwd=R,env=os.environ,text=True,capture_output=True,timeout=180)
  (B/(name+'-build.stdout')).write_text(build.stdout);(B/(name+'-build.stderr')).write_text(build.stderr)
  if build.returncode: raise RuntimeError('shader build failed: '+build.stderr[-2000:])
  shader=R/'build/shaders/MetalRobo.metallib'; sha=hashlib.sha256(shader.read_bytes()).hexdigest()
  cmd=[str(R/'build/bin/metalrobo_task_rollout'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15',
    '--birdflow-journey-teacher','--birdflow-journey-student-authority','0','--birdflow-navigation-course','training',
    '--envs','16','--steps','1400','--repeats','1','--chunk','1','--physics-substeps','4',
    '--seed','20261111','--no-scheduled-resets','--minimum-difficulty-band','10','--maximum-difficulty-band','10',
    '--policy-pack',str(policy),'--visual-observation-config','assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json']
  run=subprocess.run(cmd,cwd=R,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},text=True,capture_output=True,timeout=300)
  (A/(name+'.stdout')).write_text(run.stdout);(A/(name+'.stderr')).write_text(run.stderr)
  try:
   report=json.loads(run.stdout);ep=report['first_episode_navigation']
   metrics={'failed_environment_steps':report['failed_environment_steps'],'waypoints':ep['maximum_waypoints_by_environment'],
       'completion_count':ep['completion_seen_count'],'physical_failure_count':ep['physical_failure_count'],
       'termination_reasons':ep['termination_reason_by_environment'],'steps':ep['steps_by_environment']}
  except (ValueError,KeyError):metrics={}
  results.append({'name':name,'forward_speed_coefficient':float(scale),'shader_sha256':sha,'returncode':run.returncode,
      'seconds_including_build':time.monotonic()-start,'argv':cmd,'metrics':metrics})
  (A/'result.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
  print(name,run.returncode,metrics,flush=True)
finally:
 metal.write_text(original)
