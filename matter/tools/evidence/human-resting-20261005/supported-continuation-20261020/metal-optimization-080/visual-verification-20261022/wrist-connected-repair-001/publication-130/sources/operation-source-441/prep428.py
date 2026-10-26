from pathlib import Path
import json,hashlib,subprocess,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'D/connected-continuation-prep-428';P.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();py='/Users/n/numi-human-prep-venv-20261005/bin/python'
failures=[]
for trial in ['postfit-backtrack-424','postfit-backtrack-427']:
 for p in sorted((D/'D'/trial/'run-001/backtrack-373').glob('pass-*/audit-130/surface-self-pairs.npy')):
  a=p.parent/'summary.json';j=json.loads(a.read_text())
  if j.get('exact_PLC_surface_pairs',0)<=0:continue
  c=p.parent.parent/'candidate.npz';failures.append(dict(candidate_path=str(c),candidate_sha256=sha(c),audit130_path=str(a),audit130_sha256=sha(a),pairs_path=str(p),pairs_sha256=sha(p)))
mp=P/'failed-trials-manifest.json';mp.write_text(json.dumps(dict(trials=failures),indent=2)+'\n')
cur=D/'D/postfit-backtrack-427/run-001/backtrack-373/pass-004';prior=D/'D/connected-continuation-prep-417/run-001/self-seeds';depth=D/'D/point-depth-refresh-428/run-001'
items={'candidate':cur/'candidate.npz','audit130':cur/'audit-130/summary.json','audit159':cur/'audit-159/report.json','prior-seed-report':prior/'report.json','prior-seed-pairs':prior/'merged-self-pair-seeds.npy','failed-trials-manifest':mp,'point-depth-npz':depth/'point-depth-rows.npz','point-depth-report':depth/'365-compatible-report.json'}
owner=D/'D/prepare_connected_continuation_inputs_417.py';a=[py,str(owner)]
for k,v in items.items():a+=['--'+k,str(v),'--'+k+'-sha256',sha(v)]
a+=['--query-dir',str(cur/'audit-130/query'),'--output-root',str(P/'run-001')]
(P/'launch.json').write_text(json.dumps(dict(argv=a,pins={str(p):sha(p)for p in list(items.values())+[owner]},failed_trials=len(failures)),indent=2)+'\n')
start=time.monotonic()
with(P/'execution.log').open('x')as f:r=subprocess.run(a,stdout=f,stderr=subprocess.STDOUT)
(P/'execution.json').write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-start),indent=2)+'\n');print(r.returncode)
