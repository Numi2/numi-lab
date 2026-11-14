from pathlib import Path
import hashlib,json,sys,zlib
P=Path(sys.argv[1]);source=P/'run.stderr.gz';chunk=40*1024*1024;h=hashlib.sha256();z=zlib.decompressobj(31);parts=[];total=0
with source.open('rb') as f:
 i=0
 while True:
  b=f.read(chunk)
  if not b:break
  part=P/('run.stderr.gz.part%03d'%i)
  with part.open('xb') as o:o.write(b)
  ph=hashlib.sha256(b).hexdigest();h.update(b);z.decompress(b);total+=len(b);parts.append({'file':part.name,'bytes':len(b),'sha256':ph});i+=1
assert z.eof and not z.unused_data
(P/'stderr-parts.json').write_text(json.dumps({'original':source.name,'bytes':total,'sha256':h.hexdigest(),'parts':parts},indent=2)+'\n');print('split verified gzip',total,h.hexdigest())
