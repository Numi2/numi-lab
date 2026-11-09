from pathlib import Path
import hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-production-integration-20261109';B=R/'build/crow-production-integration-20261109'
prior=R/'artifacts/crow-terminal-specialist-paired-eval-20261109/training/results.json'
report=json.loads(prior.read_text());cmd=report['argv'].copy()
for key,value in [('--birdflow-journey-variant','v15-hierarchical-navigation'),('--metallib',str(R/'build/shaders/MetalRobo.metallib')),('--state-trace',str(B/'production.tsv'))]:
 cmd[cmd.index(key)+1]=value
env={**os.environ,'DYLD_LIBRARY_PATH':str(R/'build/lib'),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=360)
(A/'production.stdout').write_text(p.stdout);(A/'production.stderr').write_text(p.stderr)
if p.returncode:raise RuntimeError(p.stderr[-5000:])
def load(path):
 with path.open() as f:
  header=f.readline();return {int(x[0]):[float(v) for v in x[1:24]] for line in f for x in [line.strip().split('\t')] if len(x)>24}
old=load(R/'build/crow-terminal-specialist-paired-eval-20261109/training-trace.tsv');new=load(B/'production.tsv')
end=report['first_episode']['steps_by_environment'][1]
keys=sorted(k for k in old if k<=end and k in new)
assert len(keys)==end,(len(keys),end)
maxerr=max(abs(a-b) for k in keys for a,b in zip(old[k],new[k]))
data=json.loads(p.stdout);first=data['first_episode_navigation']
oldfirst=report['first_episode']
result={'argv':cmd,'steps_compared':len(keys),'q_max_absolute_difference':maxerr,'q_exact':all(old[k]==new[k] for k in keys),'first_episode_results_equal':first==oldfirst,'production_first_episode':first,'policy_pack_sha256':hashlib.sha256(Path(cmd[cmd.index('--policy-pack')+1]).read_bytes()).hexdigest(),'base_policy_sha256':hashlib.sha256(Path(cmd[cmd.index('--base-policy-pack')+1]).read_bytes()).hexdigest(),'metallib_sha256':hashlib.sha256(Path(cmd[cmd.index('--metallib')+1]).read_bytes()).hexdigest(),'runtime_sha256':hashlib.sha256((R/'build/lib/libmetalrobo.dylib').read_bytes()).hexdigest()}
(A/'production-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print('steps',len(keys),'max q difference',maxerr,'first episode results identical',result['first_episode_results_equal'],flush=True)
if maxerr>1e-6 or not result['first_episode_results_equal']:raise RuntimeError('Production physical parity failed')
