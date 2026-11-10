from pathlib import Path
import hashlib,json,sys,tarfile
root=Path(sys.argv[1]);names=sys.argv[2:];h=lambda d:hashlib.sha256(d).hexdigest();total=0
for name in names:
 a=root/"artifacts"/name
 for row in (a/"SHA256SUMS").read_text().splitlines():
  digest,file=row.split("  ",1);p=a/file
  if h(p.read_bytes())!=digest:raise SystemExit("digest mismatch "+str(p))
  total+=1
 index=a/"runtime-index.json";tar=a/"runtime.tar.gz"
 if index.exists() and tar.exists():
  with tarfile.open(tar,"r:gz") as f:
   for entry in json.loads(index.read_text()):
    m=f.extractfile(entry["path"])
    if not m or h(m.read())!=entry["sha256"]:raise SystemExit("runtime mismatch "+str(a)+" "+entry["path"])
 print(name,"verified",flush=True)
print("verified_files",total,"artifacts",len(names),flush=True)
