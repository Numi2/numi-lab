from pathlib import Path
import gzip
A=Path('artifacts/crow-v15-combined-clearance-kinematics-20261114')
with gzip.open('artifacts/crow-v15-pitch-clearance-combined-20261114/trace.log.gz','rt') as f,(A/'input.tsv').open('w') as o:
 for l in f:
  x=l.split()
  if x and x[0]=='step' and int(x[2]) in (240,250,260,265):
   q=x.index('q');v=x.index('v');o.write(' '.join([x[1],x[2]]+x[q+1:v]+x[v+1:])+'\n')
