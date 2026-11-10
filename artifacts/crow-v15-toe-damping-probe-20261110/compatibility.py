from pathlib import Path
import hashlib,json,os,subprocess
R=Path.cwd(); A=R/'artifacts/crow-v15-toe-damping-probe-20261110'; H=R/'build/crow-v15-toe-damping-probe-20261110'; B=R/'build/crow-v15-long-perch-host-20261110'
cmd=json.loads((R/'artifacts/crow-v15-integrated-route-baseline-20261110/contract-bindings.json').read_text())['argv'][:]
for k,v in [('--metallib',str(R/'build/crow-v15-student-landing-controller-20261110/teacher.metallib')),('--birdflow-navigation-course','reserved-final-j'),('--envs','1'),('--seed','3193706597'),('--steps','2'),('--policy-pack',str(B/'landing.policypack')),('--base-policy-pack',str(R/'build/crow-v15-stage-aware-learning-20261110/corrective-actor.policypack')),('--state-trace',str(H/'compatibility.tsv')),('--state-trace-environment','0'),('--crow-replay-pack',str(H/'compatibility.json'))]:cmd[cmd.index(k)+1]=v
env={**os.environ,'DYLD_LIBRARY_PATH':str(H),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,capture_output=True,text=True)
(A/'compatibility.stdout').write_text(p.stdout);(A/'compatibility.stderr').write_text(p.stderr)
try: out=json.loads(p.stdout)
except ValueError:out=None
result={'argv':cmd,'candidate_host_sha256':hashlib.sha256((H/'libmetalrobo.dylib').read_bytes()).hexdigest(),'returncode':p.returncode,'json_output':isinstance(out,dict),'note':'Two-step development binding smoke test, no contract checks disabled; compatibility does not qualify physics.'}
(A/'compatibility-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
raise SystemExit(p.returncode if p.returncode else (0 if isinstance(out,dict) else 1))
