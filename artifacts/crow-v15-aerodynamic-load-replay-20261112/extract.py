from pathlib import Path
import gzip,json,hashlib
A=Path('artifacts/crow-v15-aerodynamic-load-replay-20261112');P=Path('artifacts/crow-v15-wing-steering-without-body-moments-20261112');rows=[];meta=[]
for arm,name in enumerate(('baseline','enhanced')):
 path=P/(name+'.stderr.gz');h=hashlib.sha256(path.read_bytes()).hexdigest();n=0
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);t=x['step'];e=x['lane']
    if t>=500 and t<=1600:
     assert len(x['q'])==41 and len(x['v'])==40
     rows.append((arm,e,t,x['q'],x['v']));n+=1
 meta.append({'arm':name,'compressed_sha256':h,'samples':n})
assert len(rows)==144 and len({(a,e,t) for a,e,t,_,_ in rows})==144
(A/'input.txt').write_text(''.join(f'{a} {e} {t} '+' '.join(map(str,q+v))+'\n' for a,e,t,q,v in sorted(rows)))
(A/'input-metadata.json').write_text(json.dumps({'inputs':meta,'limits':'Sparse control endpoints, not complete microstep force histories. State comes from previously used development-seed flight episodes.'},indent=2)+'\n')
print(len(rows),'samples')
