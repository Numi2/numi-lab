from pathlib import Path
import gzip,hashlib,json,sys,os
p=json.loads(Path(sys.argv[1]).read_text())
for r in p['files']:
 a=Path(r['archive']);o=Path(r['original']);assert hashlib.sha256(a.read_bytes()).hexdigest()==r['archive_sha256'];assert not o.exists();o.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(a,'rb') as src,o.open('xb') as out:
  for b in iter(lambda:src.read(1024**2),b''):out.write(b)
  out.flush();os.fsync(out.fileno())
 assert o.stat().st_size==r['bytes'] and hashlib.sha256(o.read_bytes()).hexdigest()==r['sha256']
print('All archived original bytes restored and verified')
