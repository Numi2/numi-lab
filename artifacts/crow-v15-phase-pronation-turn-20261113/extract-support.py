from pathlib import Path
import gzip,json,sys
src=Path(sys.argv[1]);dst=Path(sys.argv[2]);states={}
with gzip.open(src,'rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);key=(x['lane'],x['step']);assert key not in states;states[key]=x
with dst.open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
  z.write(json.dumps([states[k] for k in sorted(states)],separators=(',',':')).encode())
print('states',len(states))
