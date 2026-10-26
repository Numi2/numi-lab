from pathlib import Path
import subprocess,time,json
D=Path('/Users/numi/Work/numi-human-connected-surface-069');out=D/'D/postfit-backtrack-418/run-001';py='/Users/n/numi-human-prep-venv-20261005/bin/python'
a=[py,str(D/'barycentric-connected-clearance-170/no-progress-backtracker-373/run_local_witness_backtracking_373.py'),str(out/'backtrack-config.json')]
start=time.time()
with (out/'execution.log').open('w')as f:r=subprocess.run(a,stdout=f,stderr=subprocess.STDOUT)
(out/'execution.json').write_text(json.dumps(dict(argv=a,returncode=r.returncode,wall_seconds=time.time()-start),indent=2)+'\n')
print(json.dumps(dict(returncode=r.returncode,wall_seconds=time.time()-start)))

