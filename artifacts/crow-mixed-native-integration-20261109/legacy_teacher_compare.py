from pathlib import Path
import hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-mixed-native-integration-20261109';B=R/'build/crow-production-integration-20261109'
cmd=json.loads((R/'artifacts/crow-terminal-specialist-paired-eval-20261109/training/results.json').read_text())['argv'].copy()
for k in ['--base-policy-pack']:
 i=cmd.index(k);del cmd[i:i+2]
for k,v in [('--birdflow-journey-variant','v15-active-head-navigation'),('--policy-pack',str(R/'build/crow-production-integration-20261109/legacy-regression.policypack')),('--envs','4'),('--steps','800'),('--seed','20270202')]:
 cmd[cmd.index(k)+1]=v
cmd += ['--birdflow-journey-teacher','--birdflow-journey-student-authority','0']
runs={}
for mode in ['legacy','production']:
 lib=B/'legacy' if mode=='legacy' else R/'build/lib'
 shader=B/'legacy/MetalRobo.metallib' if mode=='legacy' else R/'build/shaders/MetalRobo.metallib'
 trace=B/(mode+'-teacher-legacy.tsv')
 for k,v in [('--metallib',str(shader)),('--state-trace',str(trace))]:cmd[cmd.index(k)+1]=v
 env={**os.environ,'DYLD_LIBRARY_PATH':str(lib),'PYTHONDONTWRITEBYTECODE':'1'}
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240)
 (A/(mode+'-teacher-legacy.stdout')).write_text(p.stdout);(A/(mode+'-teacher-legacy.stderr')).write_text(p.stderr)
 if p.returncode:raise RuntimeError(p.stderr[-5000:])
 with trace.open() as f:
  header=f.readline();rows={int(x[0]):[float(v) for v in x[1:24]] for line in f for x in [line.strip().split('\t')] if len(x)>24}
 runs[mode]={'q':rows,'first':json.loads(p.stdout)['first_episode_navigation'],'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256((lib/'libmetalrobo.dylib').read_bytes()).hexdigest()}
a=runs['legacy'];b=runs['production'];keys=sorted(set(a['q'])&set(b['q']))
maxerr=max(abs(i-j) for k in keys for i,j in zip(a['q'][k],b['q'][k]))
report={'steps':len(keys),'q_max_absolute_difference':maxerr,'q_exact':all(a['q'][k]==b['q'][k] for k in keys),'first_episode_equal':a['first']==b['first'],'legacy_shader_sha256':a['shader_sha256'],'legacy_library_sha256':a['library_sha256'],'production_shader_sha256':b['shader_sha256'],'production_library_sha256':b['library_sha256']}
(A/'legacy-teacher-comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if len(keys)!=800 or maxerr>1e-5 or a['first']!=b['first']:raise RuntimeError('legacy physical behavior differs')
