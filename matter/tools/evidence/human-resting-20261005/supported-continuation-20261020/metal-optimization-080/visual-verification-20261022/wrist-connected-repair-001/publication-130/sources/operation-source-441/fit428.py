from pathlib import Path
import json,hashlib,subprocess,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';P=D/'D/surface-witness-basis-428/run-001-launch';P.mkdir(parents=True,exist_ok=False)
a=json.loads((D/'D/surface-witness-basis-417/run-001-launch/argv.json').read_text());cur=D/'D/postfit-backtrack-427/run-001/backtrack-373/pass-004/candidate.npz';prep=D/'D/connected-continuation-prep-428/run-001'
updates={'--base-candidate':cur,'--raw-trial':cur,'--area-reference-npz':cur,'--self-pairs':prep/'self-seeds/merged-self-pair-seeds.npy','--point-target-npz':prep/'point-targets-390/nearest-union-exits.npz','--point-target-report':prep/'point-targets-390/nearest-union-exits.json','--face-target-npz':prep/'surface-targets-335/targets.npz','--face-target-report':prep/'surface-targets-335/report.json','--point-depth-rows-npz':D/'D/point-depth-refresh-428/run-001/point-depth-rows.npz','--point-depth-rows-report':D/'D/point-depth-refresh-428/run-001/365-compatible-report.json','--fixed-boundary-npz':D/'D/fixed-bone-guard-refresh-428/run-001/introduced-skin-bone-barriers.npz','--fixed-boundary-report':D/'D/fixed-bone-guard-refresh-428/run-001/report.json','--output':H/'surface-witness-basis-428/run-001'}
for k,v in updates.items():a[a.index(k)+1]=str(v)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={str(p):sha(p)for p in updates.values()if Path(p).is_file()};pins[a[1]]=sha(a[1]);pins[str(prep/'preparation-report.json')]=sha(prep/'preparation-report.json')
(P/'argv.json').write_text(json.dumps(a,indent=2)+'\n');(P/'preflight.json').write_text(json.dumps(dict(pins=pins,variant='unchanged408 refreshed427 control',native_admission=False,accepted_physical_steps=0),indent=2)+'\n')
start=time.monotonic()
with(P/'execution.log').open('x')as f:r=subprocess.run([a[0],'/tmp/numi-human-geometry-slot.py',*a],stdout=f,stderr=subprocess.STDOUT)
assert all(sha(p)==h for p,h in pins.items());(P/'execution.json').write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-start),indent=2)+'\n');print(r.returncode)
