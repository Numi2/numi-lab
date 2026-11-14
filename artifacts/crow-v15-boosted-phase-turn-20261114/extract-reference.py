from pathlib import Path
import gzip,json,sys
p=Path(sys.argv[1]);out=Path(sys.argv[2]);rows=[];seen=set()
with gzip.open(p,'rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);k=x['lane'],x['step'];assert k not in seen;seen.add(k);rows.append(x)
with out.open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(json.dumps(rows,separators=(',',':')).encode())
print('states',len(rows))
