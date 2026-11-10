from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-anchored-paired-eval-20261110');out={}
for name in ('j-hard','i-hard','j-original','i-original'):
 prev=0.;last=-1;events=[]
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_TASK_STATE '):
    z=l.split();t=int(z[2]);phase=float(z[7])
    if t<last:break
    last=t
    if prev>=1.5 and phase<1.5:events.append({'step':t,'previous_phase':prev,'new_phase':phase,'left_load':float(z[5]),'right_load':float(z[6]),'height_target':float(z[8]),'forward_velocity':float(z[9])})
    prev=phase
 out[name]=events
Path('artifacts/crow-v15-retry-climb-20261110/loss-events.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
