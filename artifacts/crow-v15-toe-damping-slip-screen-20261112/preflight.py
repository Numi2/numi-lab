from pathlib import Path
import collections,json,os,re,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-toe-damping-slip-screen-20261112';B=R/'build/crow-v15-toe-damping-slip-screen-20261112'
D=R/'artifacts/crow-v15-toe-preload-matched-trial-20261112/candidate-20261127-rebound-low'
cmd=json.loads((D/'command.json').read_text())
for key,val in [('--steps','2'),('--state-trace',str(B/'preflight.tsv')),('--crow-replay-pack',str(B/'preflight-replay.json'))]:cmd[cmd.index(key)+1]=val
(A/'preflight-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_EXPERIMENT_GAIT_SMOOTH':'1','CROW_VERIFY_TOE_DAMPING':'1','PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=180)
(A/'preflight.stdout').write_text(p.stdout);(A/'preflight.stderr').write_text(p.stderr)
found=re.findall(r'^CROW_TOE_DAMPING (\S+) ([0-9.]+)$',p.stderr,re.M)
values={name:float(x) for name,x in found}
assert len(values)==16 and all(abs(v-.03)<1e-6 for v in values.values()),values
m=re.search(r'policy=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\) task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)',p.stderr)
assert p.returncode!=0 and m,'expected changed-contract policy mismatch: '+p.stderr[-2000:]
(A/'preflight-result.json').write_text(json.dumps({'return_code':p.returncode,'verified_toe_damping':values,'old_fingerprints':list(map(int,m.groups()[:4])),'new_fingerprints':list(map(int,m.groups()[4:]))},indent=2)+'\n')
print('verified',len(values),'toe drives and expected fingerprint mismatch')
