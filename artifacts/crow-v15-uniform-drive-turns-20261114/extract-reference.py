from pathlib import Path
import gzip,json
src=Path('artifacts/crow-v15-uniform-wing-drive-20261114/straight-20261126-rebound-uniform/run.stderr.gz')
out=Path('artifacts/crow-v15-uniform-drive-turns-20261114/straight-reference-states.json.gz')
a=[]
with gzip.open(src,'rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):a.append(json.loads(line.split(' ',1)[1]))
assert len(a)==14408
with out.open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write(json.dumps(a,separators=(',',':')).encode())
print(len(a))
