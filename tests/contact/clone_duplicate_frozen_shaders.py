#!/usr/bin/env python3
"""Replace exact duplicate task-owned frozen shaders with APFS clones.

Clone-on-write keeps the same independent paths and exact bytes; it does not
alter or remove research inputs. Only the two predeclared hash groups below
are touched. Never run while a listed input is open.
"""
import hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
e=root/"docs/evidence/midpoint-first-failure-20261109"
b=Path("/Users/numi/numi-matter-midpoint-compensated-build-20261109")
groups=[("eebb5247813e6f510efedd6ef5f2d0d7a3bea840cb38966b3f2db48735ddb97f",["evidence-be-precision-005","evidence-be-precision-001","evidence-be-precision-002","evidence-be-precision-003","evidence-be-precision-004","evidence-stage-native-001"]),
 ("2d90892820a64627878189a82175e149084924d92ff2c486bbf3f0de87757148",["evidence-rich-204","evidence-capture-204"])]
def digest(f):
 h=hashlib.sha256()
 with f.open("rb") as r:
  for chunk in iter(lambda:r.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
paths=[b/d/"NumiMatter.metallib" for _,names in groups for d in names]
for expected,names in groups:
 for d in names:
  f=b/d/"NumiMatter.metallib"
  if f.is_symlink() or not f.is_file() or digest(f)!=expected:raise SystemExit("frozen shader mismatch: "+str(f))
  if subprocess.run(["lsof","-t",str(f)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=1:raise SystemExit("frozen shader may be open")
before=shutil.disk_usage(b).free;changed=[]
for expected,names in groups:
 source=b/names[0]/"NumiMatter.metallib"
 for d in names[1:]:
  target=b/d/"NumiMatter.metallib";old=target.stat()
  fd,name=tempfile.mkstemp(prefix=".shader-clone-",dir=target.parent);os.close(fd);os.unlink(name)
  try:
   subprocess.run(["cp","-pc",str(source),name],check=True)
   clone=Path(name)
   if digest(clone)!=expected or digest(target)!=expected:raise RuntimeError("clone bytes changed")
   current=target.stat()
   if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)!=(old.st_dev,old.st_ino,old.st_size,old.st_mtime_ns):raise RuntimeError("target changed")
   os.replace(clone,target)
   if digest(target)!=expected:raise RuntimeError("post-replacement mismatch")
   changed.append({"path":str(target),"sha256":expected,"reference":str(source)})
  finally:Path(name).unlink(missing_ok=True)
after=shutil.disk_usage(b).free
result={"scope":"exact hash-verified task-owned frozen shader clones; paths and bytes preserved; APFS copy-on-write","replacements":changed,"free_before_bytes":before,"free_after_bytes":after,"observed_change_bytes":after-before}
out=e/"frozen-shader-clone-001.json"
if out.exists():raise SystemExit("result exists")
out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
