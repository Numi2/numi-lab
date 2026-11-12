from pathlib import Path
import gzip,json,collections
p=Path('artifacts/crow-v15-toe-preload-second-seed-20261112');out={}
for arm in ('zero','low'):
 caps=collections.Counter();support=collections.Counter();vcount=collections.Counter()
 with gzip.open(p/('candidate-20261126-rebound-'+arm)/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=200 and float(z[8])<=1e-7 and float(z[9])<=1e-7:support[e, '0_2' if t<=100 else '2_4']+=1
   elif line.startswith('CROW_TOE_V_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=200:
     vcount[e]+=1
     if max(abs(float(x)) for x in z[4:])>=11.99:caps[e,'0_2' if t<=100 else '2_4']+=1
 assert all(vcount[e]==3200 for e in range(8));out[arm]=[{'lane':e,'unloaded_0_2':support[e,'0_2'],'unloaded_2_4':support[e,'2_4'],'toe_cap_0_2':caps[e,'0_2'],'toe_cap_2_4':caps[e,'2_4']} for e in range(8)]
(p/'preparation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
