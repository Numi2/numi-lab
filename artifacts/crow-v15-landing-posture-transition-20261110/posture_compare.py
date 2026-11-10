from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-landing-posture-transition-20261110');cases={'baseline':Path('artifacts/crow-v15-rear-toe-bounce-release-20261110/diagnostic-reserved-final-j-original.stderr.gz'),'soft':A/'diagnostic-reserved-final-j-soft-original.stderr.gz','hold':A/'diagnostic-reserved-final-j-hold-original.stderr.gz'};out={}
for name,p in cases.items():
 out[name]={}
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);s=d['step']
    if s in (2450,2460,2464,2470,2480,2490,2500,2520,2540,2600):out[name][s]={'left_leg_q':d['q'][14:17],'right_leg_q':d['q'][17:20],'body_v':d['v'][:6]}
(A/'posture-compare.json').write_text(json.dumps(out,indent=2)+'\n')
for k,v in out.items():print(k,'step2470',v.get(2470,{}).get('left_leg_q'))
