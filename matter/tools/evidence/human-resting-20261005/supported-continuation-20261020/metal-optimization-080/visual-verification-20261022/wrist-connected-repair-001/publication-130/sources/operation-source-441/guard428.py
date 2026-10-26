from pathlib import Path
import json,hashlib,subprocess,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'D/fixed-bone-guard-refresh-428';P.mkdir(exist_ok=False);py='/Users/n/numi-human-prep-venv-20261005/bin/python'
base=D/'D/postfit-backtrack-415/run-001/backtrack-373/pass-002';cur=D/'D/postfit-backtrack-427/run-001/backtrack-373/pass-004';old=D/'D/fixed-bone-guard-refresh-417/run-001'
owner=D/'D/fixed-bone-guard-refresh-413/refresh_fixed_bone_guards.py'
a=[py,'/tmp/numi-human-geometry-slot.py',py,str(owner),'--previous-candidate',str(base/'candidate.npz'),'--previous-surface-report',str(base/'audit-159/report.json'),'--prior-guard-report',str(old/'report.json'),'--prior-guard-npz',str(old/'introduced-skin-bone-barriers.npz'),'--candidate',str(cur/'candidate.npz'),'--surface-report',str(cur/'audit-159/report.json'),'--output-dir',str(P/'run-001')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={str(p):sha(p)for p in [owner,base/'candidate.npz',base/'audit-159/report.json',cur/'candidate.npz',cur/'audit-159/report.json',old/'report.json',old/'introduced-skin-bone-barriers.npz']}
(P/'launch.json').write_text(json.dumps(dict(argv=a,pins=pins),indent=2)+'\n');start=time.monotonic()
with(P/'execution.log').open('x')as f:r=subprocess.run(a,stdout=f,stderr=subprocess.STDOUT)
assert all(sha(p)==h for p,h in pins.items());(P/'execution.json').write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-start),indent=2)+'\n');print(r.returncode)
