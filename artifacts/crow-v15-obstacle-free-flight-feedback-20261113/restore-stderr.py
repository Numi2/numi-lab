from pathlib import Path
import json,hashlib,os
A=Path(__file__).resolve().parent;P=A/'push-wing-20261126-rebound-feedback';m=json.loads((P/'stderr-parts.json').read_text());dest=P/m['original_name']
if dest.exists():
 assert dest.stat().st_size==m['original_size'] and hashlib.sha256(dest.read_bytes()).hexdigest()==m['original_sha256'];print('Existing gzip verified:',dest);raise SystemExit(0)
tmp=P/(m['original_name']+'.restoring');h=hashlib.sha256();n=0
with tmp.open('xb') as out:
 for x in m['parts']:
  p=P/x['name'];assert p.stat().st_size==x['size'];part=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):part.update(b);h.update(b);n+=len(b);out.write(b)
  assert part.hexdigest()==x['sha256']
assert n==m['original_size'] and h.hexdigest()==m['original_sha256'];os.replace(tmp,dest);print('Restored and verified:',dest)
