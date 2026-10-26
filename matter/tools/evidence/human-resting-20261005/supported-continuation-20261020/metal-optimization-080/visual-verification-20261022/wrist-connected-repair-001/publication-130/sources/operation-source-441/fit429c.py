from pathlib import Path
import json,hashlib,subprocess,time
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
P=D/"D/whole-face-fit-429/run-003-launch";P.mkdir(parents=True,exist_ok=False)
a=json.loads((D/"D/surface-witness-basis-428/run-001-launch/argv.json").read_text())
a[1]=str(D/"D/D/whole-face-objective-429/run_whole_face_fit_429.py")
a[a.index("--output")+1]=str(D/"barycentric-connected-clearance-170/whole-face-fit-429/run-003")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p)for p in [a[1],D/"D/D/whole-face-objective-429/whole_face_objective_429.py"]}
(P/"argv.json").write_text(json.dumps(a,indent=2)+chr(10));(P/"preflight.json").write_text(json.dumps(dict(pins=pins,matched_control="428",native_admission=False),indent=2)+chr(10))
start=time.monotonic()
with(P/"execution.log").open("x")as f:r=subprocess.run([a[0],"/tmp/numi-human-geometry-slot.py",*a],stdout=f,stderr=subprocess.STDOUT)
assert all(sha(p)==h for p,h in pins.items())
(P/"execution.json").write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-start),indent=2));print(r.returncode)
