#!/usr/bin/env python3
"""Archive a stopped producer log, verify restoration, keep the original."""
import gzip,hashlib,json,os,shutil,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/"docs/evidence/midpoint-first-failure-20261109"
name=sys.argv[1]
if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):raise SystemExit("invalid prefix")
src=p/(name+".log");execution=p/(name+"-execution.json");dst=p/(name+".log.gz");manifest=p/(name+"-log-archive.json")
if dst.exists() or manifest.exists():raise SystemExit("archive exists")
r=json.loads(execution.read_text())
if type(r.get("returncode")) is not int or "stopped_reason" not in r:raise SystemExit("producer not known to have exited")
if not src.is_file() or src.stat().st_size==0:raise SystemExit("missing log")
required=5*1024**3//2+src.stat().st_size
if shutil.disk_usage(p).free<required:raise SystemExit("free-space guard")
def identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
before=identity(src.stat());h=hashlib.sha256();fd,temp=tempfile.mkstemp(prefix=".log-archive-",dir=p);temp=Path(temp)
try:
 with os.fdopen(fd,"wb") as raw,src.open("rb") as inp:
  with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as out:
   for chunk in iter(lambda:inp.read(1024*1024),b""):h.update(chunk);out.write(chunk)
  raw.flush();os.fsync(raw.fileno())
 if identity(src.stat())!=before:raise RuntimeError("log changed while archiving")
 restored=hashlib.sha256();size=0
 with gzip.open(temp,"rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):restored.update(chunk);size+=len(chunk)
 if restored.digest()!=h.digest() or size!=before[2] or identity(src.stat())!=before:raise RuntimeError("restore check failed")
 data={"schema":"numi.completed-log-archive.v1","source":src.name,"source_sha256":h.hexdigest(),"source_bytes":before[2],"archive":dst.name,"archive_sha256":digest(temp),"archive_bytes":temp.stat().st_size,"execution_sha256":digest(execution),"producer_returncode":r["returncode"],"producer_stopped_reason":r["stopped_reason"],"decompressed_sha256_verified":True,"original_retained":True,"free_space_minimum_before_bytes":required}
 os.link(temp,dst)
 with manifest.open("x") as f:json.dump(data,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps(data,indent=2,sort_keys=True))
finally:temp.unlink(missing_ok=True)
