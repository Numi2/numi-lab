#!/usr/bin/env python3
"""Retire only the two stopped-run local originals after independent restore."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/"docs/evidence/midpoint-first-failure-20261109"
verification=json.loads((p/"be-precision-005-remote-restore-001.json").read_text())
if not verification["remote_download_and_restore_verified"] or verification["commit"]!="0155fcd4e":raise SystemExit("missing remote verification")
if subprocess.run(["git","cat-file","-e","2ab313072^{commit}"],cwd=root).returncode:raise SystemExit("missing committed restore record")
prefix="be-precision-005-n2-h00003125"
items=[]
for ext,kind in ((".json","-archive.json"),(".log","-log-archive.json")):
 path=p/(prefix+ext);manifest=json.loads((p/(prefix+kind)).read_text())
 if path.is_symlink() or not path.is_file() or path.name!=manifest["source"]:raise SystemExit("unexpected source")
 if not any(x["archive"]==manifest["archive"] and x["restored_sha256"]==manifest["source_sha256"] and x["restored_bytes"]==manifest["source_bytes"] for x in verification["records"]):raise SystemExit("remote verification mismatch")
 if subprocess.run(["git","ls-files","--error-unmatch",str(path.relative_to(root))],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise SystemExit("source is tracked")
 if subprocess.run(["lsof","-t",str(path)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=1:raise SystemExit("source may be open")
 stat=path.stat();h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 if h.hexdigest()!=manifest["source_sha256"] or stat.st_size!=manifest["source_bytes"]:raise SystemExit("source changed")
 items.append((path,stat,manifest))
before=shutil.disk_usage(p).free
for path,stat,manifest in items:
 current=path.stat()
 if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)!=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns):raise SystemExit("source changed before retirement")
 path.unlink()
after=shutil.disk_usage(p).free
result={"scope":"two untracked stopped-run originals only; remotely downloaded archives restored with exact hashes before retirement","remote_restore_record":"be-precision-005-remote-restore-001.json","items":[{"source":x.name,"bytes":s.st_size,"sha256":m["source_sha256"],"remote_archive":m["archive"]} for x,s,m in items],"free_before_bytes":before,"free_after_bytes":after,"observed_change_bytes":after-before}
out=p/"be-precision-005-local-retirement-001.json"
if out.exists():raise SystemExit("retirement record already exists")
out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
