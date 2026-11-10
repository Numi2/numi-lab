from pathlib import Path
import json,os,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-substep-contract-20261110';H=R/'build/crow-v15-substep-contract-20261110';B=R/'build/crow-v15-long-perch-host-20261110'
cmd=json.loads((R/'artifacts/crow-v15-second-clearance-eval-20261110/second-j-original-bindings.json').read_text())['argv'][:]
for k,v in [('--steps','2'),('--state-trace',str(H/'audit.tsv')),('--crow-replay-pack',str(H/'audit.json'))]:cmd[cmd.index(k)+1]=v
env={**os.environ,'DYLD_LIBRARY_PATH':str(H),'CROW_CONTRACT_CHECK':'1'}
p=subprocess.run(cmd,env=env,cwd=R,capture_output=True,text=True);(A/'audit.stdout').write_text(p.stdout);(A/'audit.stderr').write_text(p.stderr)
lines=[x for x in p.stderr.splitlines() if x.startswith('CROW_SUBSTEP_CONTRACT ')];print('\n'.join(lines));print('returncode',p.returncode)
(A/'audit.json').write_text(json.dumps({'argv':cmd,'returncode':p.returncode,'diagnostic_host_sha256':hashlib.sha256((H/'libmetalrobo.dylib').read_bytes()).hexdigest(),'records':lines},indent=2)+'\n')
raise SystemExit(p.returncode or (0 if len(lines)==3 else 2))
