from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-stance-friction-sensitivity-20261114');P=Path('artifacts/crow-v15-toe-friction-observer-20261114');out={}
for name,path in (('baseline',P/'substeps.log.gz'),('high',A/'substeps.log.gz')):
 c=collections.Counter()
 with gzip.open(path,'rt') as f:
  for l in f:
   z=l.split()
   if z and z[0]=='CROW_TOE_FRICTION':c[tuple(z[-2:])]+=1
 out[name]=[{'static':float(k[0]),'dynamic':float(k[1]),'contact_rows':v} for k,v in sorted(c.items())]
(A/'friction-coefficients.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
