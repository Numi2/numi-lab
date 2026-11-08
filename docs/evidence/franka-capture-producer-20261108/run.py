#!/usr/bin/env python3
import datetime,hashlib,subprocess,tarfile
from pathlib import Path
d=Path(__file__).resolve().parent;b=Path("/Users/numi/numi-franka-capture-producer-build-20261108")
names=("preflight.sha256-check","postflight.sha256-check","gpu-test.stdout","gpu-test.stderr","gpu-test.timing","visible.bin","retry.bin","empty.bin","raw.tar.gz","comparison.txt")
if any((d/n).exists() for n in names):raise SystemExit("refusing overwrite")
with (d/names[0]).open("w") as f:
 if subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode:raise SystemExit("preflight failed")
start=datetime.datetime.now(datetime.timezone.utc)
try:
 with (d/"gpu-test.stdout").open("w") as out,(d/"gpu-test.stderr").open("w") as err:
  rc=str(subprocess.run([str(b/"probe"),str(b/"WorldRenderer.metallib"),str(d)],cwd=d,stdout=out,stderr=err,timeout=120).returncode)
except subprocess.TimeoutExpired:rc="timeout"
end=datetime.datetime.now(datetime.timezone.utc)
(d/"gpu-test.timing").write_text(f"start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={rc}\n")
a=d/"visible.bin";c=d/"retry.bin"
if a.exists() and c.exists():
 x=a.read_bytes();y=c.read_bytes();x=x[8:];y=y[8:];diff=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),None)
 if diff is None and len(x)!=len(y):diff=min(len(x),len(y))
 (d/"comparison.txt").write_text(f"visible_bytes={len(x)} retry_bytes={len(y)} visible_sha256={hashlib.sha256(x).hexdigest()} retry_sha256={hashlib.sha256(y).hexdigest()} first_diff={diff} match={bool(x) and x==y}\n")
with (d/names[1]).open("w") as f:post=subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
with tarfile.open(d/"raw.tar.gz","w:gz") as tar:
 for n in ("visible.bin","retry.bin","empty.bin"):
  if (d/n).exists():tar.add(d/n,arcname=n)
print(f"exit={rc} start_utc={start.isoformat()} end_utc={end.isoformat()} postflight={post}")
