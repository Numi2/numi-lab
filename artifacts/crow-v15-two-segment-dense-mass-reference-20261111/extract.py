from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-two-segment-dense-mass-reference-20261111')
path=Path('artifacts/crow-v15-two-segment-frozen-flight-envelope-20261111/run.stderr.gz')
wanted={(s,e) for s in (3,100,1000,1500) for e in (0,16,32,48,64)}
out={}
with gzip.open(path,'rt') as f:
 for l in f:
  if l.startswith('CROW_FLIGHT_SAMPLE '):
   x=json.loads(l[len('CROW_FLIGHT_SAMPLE '):]);key=x['step'],x['lane']
   if key in wanted:out[key]=x['q']
if set(out)!=wanted:raise ValueError(f'missing {wanted-set(out)}')
with (A/'states.txt').open('w') as f:
 for key,q in sorted(out.items()):f.write(f'{key[0]} {key[1]} '+ ' '.join(map(str,q))+'\n')
