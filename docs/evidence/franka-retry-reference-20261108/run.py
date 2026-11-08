#!/usr/bin/env python3
import datetime,hashlib,subprocess,tarfile
from pathlib import Path
d=Path(__file__).resolve().parent
b=Path("/Users/numi/numi-franka-retry-reference-build-20261108")
fields=("q","v","scene","actor","actor_history","depth_history","track_history","camera","metadata","status","winners")
names=("preflight.sha256-check","postflight.sha256-check","comparison.tsv","raw.tar.gz")
if any((d/n).exists() for n in names) or any(d.glob("control.*")) or any(d.glob("retry.*")): raise SystemExit("refusing overwrite")
with (d/names[0]).open("w") as f:
 if subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode: raise SystemExit("preflight failed")
statuses={}
for arm in ("control","retry"):
 start=datetime.datetime.now(datetime.timezone.utc)
 try:
  with (d/(arm+".stdout")).open("w") as out,(d/(arm+".stderr")).open("w") as err:
   rc=str(subprocess.run([str(b/"probe"),str(b/"WorldRenderer.metallib"),str(b/"pick_object.mrvpack"),str(d/(arm+".snapshot")),arm],cwd=d,stdout=out,stderr=err,timeout=120).returncode)
 except subprocess.TimeoutExpired: rc="timeout"
 end=datetime.datetime.now(datetime.timezone.utc);statuses[arm]=rc
 (d/(arm+".timing")).write_text(f"start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={rc}\n")
 print(f"{arm} exit={rc} start_utc={start.isoformat()} end_utc={end.isoformat()}",flush=True)
 if rc!="0": break
with (d/"comparison.tsv").open("w") as f:
 f.write("field\tcontrol_bytes\tretry_bytes\tcontrol_sha256\tretry_sha256\tfirst_diff\tresult\n")
 for field in fields:
  a=d/("control.snapshot."+field);c=d/("retry.snapshot."+field)
  if not a.exists() or not c.exists():
   f.write(f"{field}\t-\t-\t-\t-\t-\tMISSING\n");continue
  x=a.read_bytes();y=c.read_bytes();diff=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),None)
  if diff is None and len(x)!=len(y):diff=min(len(x),len(y))
  verdict="MATCH" if x and y and diff is None else "DIFF_OR_EMPTY"
  f.write(f"{field}\t{len(x)}\t{len(y)}\t{hashlib.sha256(x).hexdigest()}\t{hashlib.sha256(y).hexdigest()}\t{diff if diff is not None else '-'}\t{verdict}\n")
with (d/names[1]).open("w") as f: post=subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
with tarfile.open(d/"raw.tar.gz","w:gz") as tar:
 for arm in ("control","retry"):
  for path in sorted(d.glob(arm+".snapshot*")): tar.add(path,arcname=path.name)
print(f"postflight={post} statuses={statuses}",flush=True)
