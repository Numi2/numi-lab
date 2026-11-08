#!/usr/bin/env python3
import datetime,hashlib,struct,subprocess,tarfile
from pathlib import Path
d=Path(__file__).resolve().parent;b=Path("/Users/numi/numi-franka-staged-publication-build-20261108")
names=("preflight.sha256-check","postflight.sha256-check","comparison.txt","raw.tar.gz")
if any((d/n).exists() for n in names) or any(d.glob("control.*")) or any(d.glob("mixed.*")):raise SystemExit("refusing overwrite")
with (d/names[0]).open("w") as f:
 if subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode:raise SystemExit("preflight failed")
statuses={}
for arm in ("control","mixed"):
 start=datetime.datetime.now(datetime.timezone.utc)
 try:
  with (d/(arm+".stdout")).open("w") as out,(d/(arm+".stderr")).open("w") as err:
   rc=str(subprocess.run([str(b/"probe"),str(b/"WorldRenderer.metallib"),str(b/"pick_object.mrvpack"),str(d/(arm+".bin")),arm],cwd=d,stdout=out,stderr=err,timeout=120).returncode)
 except subprocess.TimeoutExpired:rc="timeout"
 end=datetime.datetime.now(datetime.timezone.utc);statuses[arm]=rc
 (d/(arm+".timing")).write_text(f"start_utc={start.isoformat()}\nend_utc={end.isoformat()}\nexit={rc}\n")
 print(f"{arm} exit={rc} start_utc={start.isoformat()} end_utc={end.isoformat()}",flush=True)
 if rc!="0": break
def parse(path):
 data=path.read_bytes();offset=0;rows=[]
 for _ in range(4):
  if offset+8>len(data):raise ValueError("short length")
  n=struct.unpack_from("<Q",data,offset)[0];offset+=8
  size=n*4
  if not n or offset+size>len(data):raise ValueError("bad row")
  rows.append(data[offset:offset+size]);offset+=size
 if offset!=len(data):raise ValueError("trailing data")
 return rows
a=d/"control.bin";c=d/"mixed.bin"
if all(statuses.get(x)=="0" for x in ("control","mixed")) and a.exists() and c.exists():
 try:
  x=parse(a);y=parse(c);match=x[1]==y[1];sentinel=y[2]==struct.pack("<f",-1234.0)*(len(y[2])//4);accepted=y[3]==y[1]
  (d/"comparison.txt").write_text(f"control_env1_bytes={len(x[1])} mixed_env1_bytes={len(y[1])} control_sha256={hashlib.sha256(x[1]).hexdigest()} mixed_sha256={hashlib.sha256(y[1]).hexdigest()} env1_match={match} env0_sentinel_retained={sentinel} env1_candidate_published={accepted}\n")
 except ValueError as e:(d/"comparison.txt").write_text(f"comparison_error={e}\n")
with (d/names[1]).open("w") as f:post=subprocess.run(["shasum","-a","256","-c","declared-identities.sha256"],cwd=d,stdout=f,stderr=subprocess.STDOUT).returncode
with tarfile.open(d/"raw.tar.gz","w:gz") as tar:
 for n in ("control.bin","mixed.bin"):
  if (d/n).exists():tar.add(d/n,arcname=n)
print(f"postflight={post} statuses={statuses}",flush=True)
