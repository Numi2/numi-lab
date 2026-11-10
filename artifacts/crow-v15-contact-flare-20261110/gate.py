from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-contact-flare-20261110');fk={}
for line in (A/'fk.log').read_text().splitlines():
 p=line.split();fk[int(p[1])]=(float(p[7]),float(p[11]))
with gzip.open(A/'development-j-contact-flare.stderr.gz','rt') as f:
 for line in f:
  if not line.startswith('CROW_TASK_STATE '):continue
  p=line.split();t=int(p[2]);nav=float(p[4]);l,r=float(p[5]),float(p[6]);phase=float(p[7])
  if nav>=5 and (l>.1 and r>.1 and l+r>2.2):
   print(t,'load',round(l,3),round(r,3),'phase',phase,'feet',fk.get(t+1),'target',p[8],'speed',p[9])
