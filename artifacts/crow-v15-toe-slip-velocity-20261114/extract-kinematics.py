from pathlib import Path
import gzip
A=Path('artifacts/crow-v15-toe-slip-velocity-20261114')
seen=set();count=0
with gzip.open(A/'substeps.log.gz','rt') as f,(A/'kinematics-input.txt').open('w') as out:
 for line in f:
  z=line.split()
  if z and z[0]=='CROW_SLIP_QV':
   assert len(z)==85,(len(z),z[:4]);step,sub,lane=map(int,z[1:4]);key=(lane,step,sub);assert key not in seen;seen.add(key)
   print(lane,step,sub,*z[4:],file=out);count+=1
assert count==252 and seen=={(e,t,s) for e in (0,1) for t in range(290,311) for s in (0,1,15,16,30,31)}
print('Complete q/v records:',count)
