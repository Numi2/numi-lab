import gzip,json
from pathlib import Path
A=Path('artifacts/crow-v15-foot-alignment-gate-20261110')
for name in ('gate','zero_bias'):
 path=A/('development-reserved-final-j-'+name+'.stderr.gz'); states={};supports={}
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_TASK_STATE '):
    v=line.split();i,t=int(v[1]),int(v[2]);
    if i==0 and t<=4500 and float(v[4])>=5 and float(v[5])>.1 and float(v[6])>.1 and float(v[5])+float(v[6])>2.2:states[t+1]=(float(v[5]),float(v[6]))
   elif line.startswith('CROW_SUPPORT_SAMPLE '):
    v=json.loads(line.split(' ',1)[1]);t=v['step'];
    if v['lane']==0 and 2800<=t<=4500:supports[t]=v
 chosen=sorted(set(states)&set(supports) | {t for t in supports if t%100==0});
 with (A/(name+'-samples.txt')).open('w') as f:
  for t in chosen:
   v=supports[t];f.write(str(t)+' '+' '.join(map(str,v['q']+v['v']))+'\n')
 (A/(name+'-sample-info.json')).write_text(json.dumps({'loaded_task_to_support_step':states,'samples':chosen},indent=2)+'\n')
