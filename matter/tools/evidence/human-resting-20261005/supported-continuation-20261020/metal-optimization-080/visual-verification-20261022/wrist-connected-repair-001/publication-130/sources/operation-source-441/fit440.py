from pathlib import Path
import json,hashlib,subprocess,time
D=Path("/Users/numi/Work/numi-human-connected-surface-069");done=D/"D/refresh-438/prep-execution.json"
t=time.monotonic()
while not done.exists():
 if time.monotonic()-t>600:raise TimeoutError("438 preparation incomplete")
 time.sleep(2)
assert json.loads(done.read_text())["returncode"]==0
P=D/"D/grouped-fit-440/run-001-launch";P.mkdir(parents=True,exist_ok=False)
a=json.loads((D/"D/reseeded-fit-434/run-001-launch/argv.json").read_text());prep=D/"D/connected-continuation-prep-438/run-001";cur=D/"D/seed-preserving-backtrack-436/run-001/backtrack-373/pass-015/candidate.npz"
a[1]=str(D/"D/D/whole-face-objective-435/run_grouped_fit_435.py")
updates={"--base-candidate":cur,"--raw-trial":cur,"--area-reference-npz":cur,"--self-pairs":prep/"self-seeds/merged-self-pair-seeds.npy","--point-target-npz":prep/"point-targets-390/nearest-union-exits.npz","--point-target-report":prep/"point-targets-390/nearest-union-exits.json","--face-target-npz":prep/"surface-targets-335/targets.npz","--face-target-report":prep/"surface-targets-335/report.json","--point-depth-rows-npz":D/"D/point-depth-refresh-438/run-001/point-depth-rows.npz","--point-depth-rows-report":D/"D/point-depth-refresh-438/run-001/365-compatible-report.json","--fixed-boundary-npz":D/"D/fixed-bone-guard-refresh-438/run-001/introduced-skin-bone-barriers.npz","--fixed-boundary-report":D/"D/fixed-bone-guard-refresh-438/run-001/report.json","--output":D/"barycentric-connected-clearance-170/grouped-fit-440/run-001"}
for k,v in updates.items():a[a.index(k)+1]=str(v)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={str(p):sha(p)for p in [a[1],D/"D/D/whole-face-objective-435/whole_face_objective_435.py",D/"D/D/whole-face-objective-429/whole_face_objective_429.py",prep/"preparation-report.json"]+[p for p in updates.values()if p.is_file()]}
(P/"argv.json").write_text(json.dumps(a,indent=2)+chr(10));(P/"preflight.json").write_text(json.dumps(dict(pins=pins,variant="one union-exit plane per face/bone; 435 on checked436",native_admission=False),indent=2)+chr(10))
t=time.monotonic()
with(P/"execution.log").open("x")as log:r=subprocess.run([a[0],"/tmp/numi-human-geometry-slot.py",*a],stdout=log,stderr=subprocess.STDOUT)
assert all(sha(p)==v for p,v in pins.items())
(P/"execution.json").write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-t),indent=2));print(r.returncode)
