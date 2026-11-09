#!/usr/bin/env python3
"""Independently download the two published stopped-run archives and restore hashes."""
import gzip,hashlib,json,os,subprocess,tempfile,urllib.request,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/"docs/evidence/midpoint-first-failure-20261109"
commit="0155fcd4e"
records=[]
for suffix in ("-archive.json","-log-archive.json"):
 manifest=json.loads((p/("be-precision-005-n2-h00003125"+suffix)).read_text())
 endpoint="repos/Numi2/numi-lab/contents/docs/evidence/midpoint-first-failure-20261109/"+manifest["archive"]+"?ref="+commit
 fd,name=tempfile.mkstemp(prefix="numi-remote-restore-",dir="/private/tmp")
 try:
  token=subprocess.check_output(["gh","auth","token"],text=True).strip()
  request=urllib.request.Request("https://api.github.com/"+endpoint,headers={
   "Authorization":"Bearer "+token,"Accept":"application/vnd.github.raw+json",
   "Accept-Encoding":"identity","User-Agent":"numi-archive-restore-check"})
  with os.fdopen(fd,"wb") as dest,urllib.request.urlopen(request,timeout=120) as response:
   shutil.copyfileobj(response,dest,length=1024*1024)
  h=hashlib.sha256()
  with open(name,"rb") as f:
   for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
  if h.hexdigest()!=manifest["archive_sha256"]:raise RuntimeError("remote archive hash mismatch")
  restored=hashlib.sha256();size=0
  with gzip.open(name,"rb") as f:
   for chunk in iter(lambda:f.read(1024*1024),b""):restored.update(chunk);size+=len(chunk)
  if restored.hexdigest()!=manifest["source_sha256"] or size!=manifest["source_bytes"]:raise RuntimeError("remote restore hash mismatch")
  records.append({"archive":manifest["archive"],"archive_sha256":h.hexdigest(),"restored_sha256":restored.hexdigest(),"restored_bytes":size})
 finally:Path(name).unlink(missing_ok=True)
result={"commit":commit,"source":"independent GitHub content API download","remote_download_and_restore_verified":True,"records":records}
out=p/"be-precision-005-remote-restore-001.json"
if out.exists():raise SystemExit("restore record already exists")
out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
