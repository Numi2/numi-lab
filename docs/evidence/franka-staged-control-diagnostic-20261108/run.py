#!/usr/bin/env python3
import datetime,subprocess,tarfile
from pathlib import Path
d=Path(__file__).resolve().parent;b=Path("/Users/numi/numi-franka-staged-control-diagnostic-build-20261108")
names=("preflight.sha256-check","postflight.sha256-check","control.stdout","control.stderr","control.timing","control.bin","raw.tar.gz")
if any((d/n).exists() for n in names):raise SystemExit("refusing overwrite")
with (d/names[0]).open("w") as f:
 if subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode:raise SystemExit("preflight failed")
start=datetime.datetime.now(datetime.timezone.utc)
try:
 with (d/"control.stdout").open("w") as out,(d/"control.stderr").open("w") as err:
  rc=str(subprocess.run([str(b/"probe"),str(b/"WorldRenderer.metallib"),str(b/"pick_object.mrvpack"),str(d/"control.bin"),"control"],cwd=d,stdout=out,stderr=err,timeout=120).returncode)
except subprocess.TimeoutExpired:rc="timeout"
end=datetime.datetime.now(datetime.timezone.utc)
(d/"control.timing").write_text(f"start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={rc}\n")
with (d/names[1]).open("w") as f:post=subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
with tarfile.open(d/"raw.tar.gz","w:gz") as tar:
 if (d/"control.bin").exists():tar.add(d/"control.bin",arcname="control.bin")
print(f"exit={rc} start_utc={start.isoformat()} end_utc={end.isoformat()} postflight={post}",flush=True)
