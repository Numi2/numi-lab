from pathlib import Path
import json,gzip
p=Path('artifacts/crow-v15-matched-nonleg-ablation-20261114');ds={}
for arm in ('push','hold'):
 x=json.load(gzip.open(p/(arm+'-window.json.gz'),'rt'));ds[arm]={k:{tuple(r[:3]):r[3:] for r in x[k]} for k in ('aero','contact')};ds[arm]['target']={tuple(r[:2]):r[2:] for r in x['target']}
out=[]
for e in range(8):
 x=ds['push']['target'];y=ds['hold']['target'];tail=next((t for t in range(335,501) if x[e,t][12]!=y[e,t][12]),None);wings=next((t for t in range(335,501) if x[e,t][6:12]!=y[e,t][6:12]),None);v={}
 for arm in ('push','hold'):
  d=ds[arm];keys=[(e,t,k) for t in range(371,377) for k in range(16)];assert all(i in d['aero'] and i in d['contact'] for i in keys);v[arm]={'vertical_aero_impulse_Ns':sum(d['aero'][i][3]*.02/16 for i in keys),'upward_foot_impulse_Ns':sum(sum(d['contact'][i][2:4]) for i in keys)}
 out.append({'lane':e,'first_wing_difference':wings,'first_tail_difference':tail,'intervals_371_to_376':v})
(p/'command-window-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
