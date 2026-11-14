from pathlib import Path
import hashlib,json,sys,zlib
A=Path('artifacts/crow-v15-flight-command-refactor-20261114');p=A/'turn-20261126-rebound-command';src=p/'run.stderr.gz';h=hashlib.sha256();z=zlib.decompressobj(31);n=0;parts=[]
with src.open('rb') as f:
 for i in range(100):
  data=f.read(40*1024*1024)
  if not data:break
  dest=p/('run.stderr.gz.part%03d'%i);assert not dest.exists();dest.write_bytes(data);h.update(data);z.decompress(data);n+=len(data);parts.append({'file':dest.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
assert n==src.stat().st_size and z.eof and not z.unused_data
m={'file':src.name,'bytes':n,'sha256':h.hexdigest(),'gzip_eof':True,'parts':parts};(p/'stderr-parts.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
