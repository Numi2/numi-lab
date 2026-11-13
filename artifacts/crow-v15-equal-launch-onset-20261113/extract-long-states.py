from pathlib import Path
import gzip,hashlib,json,sys
source=Path(sys.argv[1]);out=Path(sys.argv[2]);h=hashlib.sha256()
with source.open('rb') as f:
 for b in iter(lambda:f.read(1<<20),b''):h.update(b)
a=[]
with gzip.open(source,'rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):a.append(json.loads(line.split(' ',1)[1]))
with out.open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(json.dumps(a,separators=(',',':')).encode())
print(json.dumps({'source_sha256':h.hexdigest(),'states':len(a)}))
