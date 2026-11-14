from pathlib import Path
import gzip,hashlib,json,subprocess,sys,zlib
A=Path('artifacts/crow-v15-height-turn-seed27-reference-20261114');D=Path('artifacts/crow-v15-height-feedback-strength-20261114/altitude-20261127-rebound-switch');manifest=json.loads((D/'stderr-parts.json').read_text());commit=sys.argv[1];whole=hashlib.sha256();z=zlib.decompressobj(31);buf=b'';states={};total=0
for part in manifest['parts']:
 proc=subprocess.Popen(['git','show',commit+':'+str(D/part['file'])],stdout=subprocess.PIPE);ph=hashlib.sha256();size=0
 for chunk in iter(lambda:proc.stdout.read(1<<20),b''):
  size+=len(chunk);total+=len(chunk);ph.update(chunk);whole.update(chunk);buf+=z.decompress(chunk);lines=buf.split(b'\n');buf=lines.pop()
  for line in lines:
   if line.startswith(b'CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(b' ',1)[1]);k=(x['lane'],x['step'])
    if 1<=x['step']<=1800:
     assert k not in states;states[k]=x
 assert proc.wait()==0 and size==part['bytes'] and ph.hexdigest()==part['sha256']
assert total==manifest['bytes'] and whole.hexdigest()==manifest['sha256'] and z.eof and not z.unused_data and len(states)>=12324
out=[states[k] for k in sorted(states)]
with (A/'reference.json.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write(json.dumps(out,separators=(',',':')).encode())
print(json.dumps({'source_commit':commit,'sha256':whole.hexdigest(),'gzip_eof':z.eof,'states':len(states),'per_lane':{str(e):sum(k[0]==e for k in states) for e in range(8)}}))
