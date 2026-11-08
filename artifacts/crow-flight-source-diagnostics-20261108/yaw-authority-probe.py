from pathlib import Path
import gzip,hashlib,json,os,subprocess,time,math
R=Path.cwd();A=R/'artifacts/crow-flight-source-diagnostics-20261108';B=R/'build/crow-flight-source-diagnostics-01'
lib=B/'flight-bench-lib/libmetalrobo.dylib';shader=B/'yaw-authority-probe.metallib';policy=B/'obstacle-free-policy.policypack'
cmd=[str(R/'build/bin/metalrobo_task_rollout'),'--birdflow-american-crow-journey','--birdflow-journey-variant','v15',
 '--birdflow-journey-teacher','--birdflow-journey-student-authority','0','--metallib',str(shader),
 '--birdflow-navigation-course','training','--envs','32','--steps','1600','--repeats','1','--chunk','1',
 '--physics-substeps','4','--seed','20261111','--no-scheduled-resets','--minimum-difficulty-band','10','--maximum-difficulty-band','10',
 '--policy-pack',str(policy),'--visual-observation-config','assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json',
 '--state-trace',str(B/'yaw-authority-probe-lane-0.tsv'),'--state-trace-environment','0']
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(lib.parent)}
t=time.monotonic();p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=300)
(A/'yaw-authority-probe.stdout').write_text(p.stdout)
with (A/'yaw-authority-probe.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write(p.stderr.encode())
if p.returncode:raise RuntimeError(p.stderr[-4000:])
data=json.loads(p.stdout);first=data['first_episode_navigation']
samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_FLIGHT_SAMPLE ')]
contacts=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_CONTACT_TRACE ')]
out=[]
for mode,name in enumerate(['straight','constant-positive-yaw','constant-negative-yaw','climb','descent']):
 lanes=[i for i in range(32) if i%5==mode]
 selected=[x for x in samples if x['lane'] in lanes and x['step']<=first['steps_by_environment'][x['lane']] and x['step']>=300]
 out.append({'mode':name,'lanes':lanes,'physical_failures':sum(first['physical_failure_by_environment'][i] for i in lanes),
 'termination_reasons':[first['termination_reason_by_environment'][i] for i in lanes],
 'first_episode_steps':[first['steps_by_environment'][i] for i in lanes],
 'sample_count':len(selected),'sampled_height_min':min((x['q'][2] for x in selected if x['q']),default=None),
 'sampled_height_max':max((x['q'][2] for x in selected if x['q']),default=None)})
report={'argv':cmd,'seconds':time.monotonic()-t,'diagnostic_shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),
 'diagnostic_library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),
 'failed_environment_steps':data['failed_environment_steps'],'first_episode':first,'per_command':out,
 'periodic_samples':len(samples),'fatal_contact_records':len(contacts),
 'scope':'Development diagnostic, not qualification: estimated v15 body, obstacles disabled, fixed teacher controls; sampled at 2s intervals. Starts use native randomization, not dedicated recovery perturbations.'}
(A/'yaw-authority-probe-results.json').write_text(json.dumps(report,indent=2)+'\n')
for x in out: print(x['mode'],'failed',x['physical_failures'],'of',len(x['lanes']),'reasons',x['termination_reasons'],'height',x['sampled_height_min'],x['sampled_height_max'],flush=True)
print('failed_simulator_steps',data['failed_environment_steps'],'samples',len(samples),flush=True)
