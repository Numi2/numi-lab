from pathlib import Path
import gzip,hashlib
A=Path(__file__).resolve().parent
trace=A.parent/'crow-v15-hiproll-grip-release-20261113/trace.log.gz'
with gzip.open(trace,'rt') as f,(A/'states.tsv').open('w') as out:
 for line in f:
  x=line.split()
  if x and x[0]=='step':
   assert x[3:6]==['required',x[4],'q'] and x[47]=='v' and len(x)==88
   out.write(' '.join([x[1],x[2],*x[6:47],*x[48:]])+'\n')
print('trace sha256',hashlib.sha256(trace.read_bytes()).hexdigest())
