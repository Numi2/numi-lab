#!/usr/bin/env python3
import datetime,gzip,shutil,subprocess
from pathlib import Path
d=Path(__file__).resolve().parent; b=Path("/Users/numi/numi-franka-frame-gate-build-20261108"); asset=Path("/Users/numi/numi-franka-frame-gate-v2-build-20261108/pick_object.mrvpack")
names=("preflight.sha256-check","postflight.sha256-check","gpu-test.stdout","gpu-test.stderr","gpu-test.timing","snapshot.bin","snapshot.bin.gz")
if any((d/n).exists() for n in names):raise SystemExit("refusing overwrite")
with (d/names[0]).open("w") as f:
 if subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode:raise SystemExit("hash failure")
start=datetime.datetime.now(datetime.timezone.utc)
try:
 with (d/"gpu-test.stdout").open("w") as out,(d/"gpu-test.stderr").open("w") as err:status=str(subprocess.run([str(b/"frame_gate_probe"),str(b/"WorldRenderer.metallib"),str(asset),str(d/"snapshot.bin")],cwd=d,stdout=out,stderr=err,timeout=120).returncode)
except subprocess.TimeoutExpired:status="timeout"
end=datetime.datetime.now(datetime.timezone.utc)
(d/"gpu-test.timing").write_text(f"start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={status}\n")
with (d/names[1]).open("w") as f:post=subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
if (d/"snapshot.bin").exists():
 with (d/"snapshot.bin").open("rb") as src,gzip.open(d/"snapshot.bin.gz","wb") as dst:shutil.copyfileobj(src,dst)
print(f"exit={status} start_utc={start.isoformat()} end_utc={end.isoformat()} postflight={post}")

