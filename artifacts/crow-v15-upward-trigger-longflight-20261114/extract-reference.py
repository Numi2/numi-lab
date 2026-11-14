from pathlib import Path
import gzip,hashlib,json,sys
p=Path(sys.argv[1]);states=[];digest=hashlib.sha256()
with p.open('rb') as raw:
 for chunk in iter(lambda:raw.read(1<<20),b''):digest.update(chunk)
h=digest.hexdigest()
with gzip.open(p,'rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):states.append(json.loads(line.split(' ',1)[1]))
keys=[(x['lane'],x['step']) for x in states];assert len(keys)==len(set(keys))
o=Path(sys.argv[2])
with o.open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(json.dumps(states,separators=(',',':')).encode())
print(json.dumps({'source_sha256':h,'states':len(states),'per_lane':{str(e):sum(x['lane']==e for x in states) for e in range(8)}}))
