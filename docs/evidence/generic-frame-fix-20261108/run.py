#!/usr/bin/env python3
import datetime, gzip, shutil, subprocess
from pathlib import Path
d=Path(__file__).resolve().parent
b=Path("/Users/numi/numi-generic-frame-fix-build-20261108")
rb=Path("/Users/numi/numi-generic-renderer-depth-20261108")
cases=[
 ("depth",["/Users/numi/numi-generic-depth-validity-build-20261108/depth_validity_probe",str(b/"HybridRenderer.metallib")]),
 ("frame",["/Users/numi/numi-generic-frame-validity-build-20261108/frame_probe",str(b/"HybridRenderer.metallib")]),
 ("renderer",[str(rb/"renderer_probe"),str(rb/"WorldRenderer.metallib"),str(rb/"Original.metallib"),str(b/"HybridRenderer.metallib"),str(d/"planes.bin")])]
for p in [d/"preflight.sha256-check",d/"postflight.sha256-check",d/"planes.bin",d/"planes.bin.gz"]+[d/(x+"."+suffix) for x,_ in cases for suffix in ("stdout","stderr","timing")]:
 if p.exists(): raise SystemExit("refusing overwrite: "+str(p))
with (d/"preflight.sha256-check").open("w") as f:
 if subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode:raise SystemExit("hash failure")
for name,cmd in cases:
 start=datetime.datetime.now(datetime.timezone.utc)
 try:
  with (d/(name+".stdout")).open("w") as out,(d/(name+".stderr")).open("w") as err: status=str(subprocess.run(cmd,cwd=d,stdout=out,stderr=err,timeout=120).returncode)
 except subprocess.TimeoutExpired: status="timeout"
 end=datetime.datetime.now(datetime.timezone.utc)
 (d/(name+".timing")).write_text(f"start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={status}\n")
 print(f"{name}: exit={status} start_utc={start.isoformat()} end_utc={end.isoformat()}",flush=True)
 if status!="0": break
with (d/"postflight.sha256-check").open("w") as f: post=subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
if (d/"planes.bin").exists():
 with (d/"planes.bin").open("rb") as src,gzip.open(d/"planes.bin.gz","wb") as dst:shutil.copyfileobj(src,dst)
print("postflight="+str(post))

