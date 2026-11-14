from pathlib import Path
import gzip
A=Path('artifacts/crow-v15-stance-kinematic-sensitivity-20261114')
with gzip.open('artifacts/crow-v15-com-stance-balance-20261114/trace.log.gz','rt') as f,(A/'input.tsv').open('w') as o:
 for line in f:
  x=line.split()
  if x and x[0]=='step' and int(x[2]) in (200,210,220,225,230,235,240):
   q=x.index('q');v=x.index('v');o.write(' '.join([x[1],x[2]]+x[q+1:v]+x[v+1:])+'\n')
