from pathlib import Path
import json,hashlib,subprocess,time
D=Path("/Users/numi/Work/numi-human-connected-surface-069");P=D/"D/reseeded-fit-434/run-001-launch";P.mkdir(parents=True,exist_ok=False)
a=json.loads((D/"D/surface-witness-basis-428/run-001-launch/argv.json").read_text());prep=D/"D/connected-continuation-prep-433/run-001"
for k,v in {"--self-pairs":prep/"self-seeds/merged-self-pair-seeds.npy","--point-target-npz":prep/"point-targets-390/nearest-union-exits.npz","--point-target-report":prep/"point-targets-390/nearest-union-exits.json","--face-target-npz":prep/"surface-targets-335/targets.npz","--face-target-report":prep/"surface-targets-335/report.json","--output":D/"barycentric-connected-clearance-170/reseeded-fit-434/run-001"}.items():a[a.index(k)+1]=str(v)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={str(p):sha(p)for p in [a[1],prep/"preparation-report.json",prep/"self-seeds/merged-self-pair-seeds.npy"]}
(P/"argv.json").write_text(json.dumps(a,indent=2)+chr(10));(P/"preflight.json").write_text(json.dumps(dict(pins=pins,variant="428 with fresh failed-pair cutting constraints",native_admission=False),indent=2)+chr(10))
t=time.monotonic()
with(P/"execution.log").open("x")as log:r=subprocess.run([a[0],"/tmp/numi-human-geometry-slot.py",*a],stdout=log,stderr=subprocess.STDOUT)
assert all(sha(p)==v for p,v in pins.items())
(P/"execution.json").write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-t),indent=2));print(r.returncode)
