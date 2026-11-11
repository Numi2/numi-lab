from pathlib import Path
import gzip,json
D=[Path('artifacts/crow-v15-landing-descent-dev-20261111/trial'),Path('artifacts/crow-v15-perch-toe-world-index-dev-20261111/trial')];rows=[];first={}
for i,d in enumerate(D):
 s={}
 for p in sorted(d.glob('stderr-*.gz')):
  try:
   with gzip.open(p,'rt') as f:
    for line in f:
     if line.startswith('CROW_SUPPORT_SAMPLE '):
      x=json.loads(line.split(' ',1)[1]);s[x['lane'],x['step']]=x
     elif i==1 and line.startswith('CROW_SCENE_CONTACT '):
      a=line.split();e=int(a[3]);first.setdefault(e,int(a[1]))
  except EOFError:pass
 rows.append(s)
for e in range(8):
 keys=sorted(t for lane,t in rows[1] if lane==e and (e,t) in rows[0]);diff=next((t for t in keys if any(rows[0][e,t][k]!=rows[1][e,t][k] for k in ('q','v','policy_action'))),None)
 maxima=max((max(abs(x-y) for x,y in zip(rows[0][e,t]['q'][23:39],rows[1][e,t]['q'][23:39])) for t in keys if t>=first.get(e,10**9)),default=0)
 print(e,'first_perch_contact',first.get(e),'first_state_difference',diff,'max_toe_angle_difference_rad',round(maxima,6))
