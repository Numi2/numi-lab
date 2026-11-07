#!/usr/bin/env python3
import datetime, subprocess, gzip, shutil
from pathlib import Path
d=Path(__file__).resolve().parent
b=Path("/Users/numi/numi-generic-renderer-depth-20261108")
names=("preflight.sha256-check","postflight.sha256-check","gpu-test.stdout","gpu-test.stderr","gpu-test.timing","planes.bin","planes.bin.gz")
if any((d/n).exists() for n in names): raise SystemExit("refusing overwrite")
with (d/names[0]).open("w") as f:
 if subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode: raise SystemExit("hash failure")
start=datetime.datetime.now(datetime.timezone.utc)
try:
 with (d/"gpu-test.stdout").open("w") as out,(d/"gpu-test.stderr").open("w") as err:
  status=str(subprocess.run([str(b/"renderer_probe"),str(b/"WorldRenderer.metallib"),str(b/"Original.metallib"),str(b/"Candidate.metallib"),str(d/"planes.bin")],cwd=d,stdout=out,stderr=err,timeout=120).returncode)
except subprocess.TimeoutExpired: status="timeout"
end=datetime.datetime.now(datetime.timezone.utc)
(d/"gpu-test.timing").write_text(f"start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={status}\n")
with (d/names[1]).open("w") as f: post=subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
if (d/"planes.bin").exists():
 with (d/"planes.bin").open("rb") as src,gzip.open(d/"planes.bin.gz","wb") as dst: shutil.copyfileobj(src,dst)
print(f"exit={status} start_utc={start.isoformat()} end_utc={end.isoformat()} postflight={post}")

