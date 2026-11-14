from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-release-clock-stroke-20261114');d={}
for arm in ('push','hold'):
 x=json.load(gzip.open(A/(arm+'-window.json.gz'),'rt'));d[arm]={'s':{(v['lane'],v['step']):v for v in x['state']},'c':{tuple(v[:3]):v[3:] for v in x['contact']},'a':{tuple(v[:3]):v[3:] for v in x['aero']}}
parity=[]
for e in range(8):
 last=368 if e in (6,7) else 500;missing=[];bad=[]
 for t in range(1,last+1):
  a=d['push']['s'].get((e,t));b=d['hold']['s'].get((e,t))
  if a is None or b is None:missing.append(t);continue
  for key in ('q','v','policy_action','contacts'):
   if a[key]!=b[key]:bad.append([t,key])
 parity.append({'lane':e,'expected_steps':last,'missing':missing,'differences':bad,'exact':not missing and not bad})
rows=[]
for arm in d:
 for e in (6,7):
  c=d[arm]['c'];a=d[arm]['a'];s=d[arm]['s'];first=next((t for t in range(371,401) if all((e,t,k) in c and sum(c[e,t,k][2:4])<=1e-7 for k in range(16))),None)
  vals=[a[e,t,k][3]*.02/16 for t in range(371,first) for k in range(16) if (e,t,k) in a] if first else []
  rows.append({'arm':arm,'lane':e,'first_unloaded_interval':first,'aero_samples':len(vals),'aero_expected':16*(first-371) if first else None,'positive_vertical_aero_impulse_Ns':sum(v for v in vals if v>0),'negative_vertical_aero_impulse_Ns':sum(v for v in vals if v<0),'net_vertical_aero_impulse_Ns':sum(vals),'max_observer_flap_motion_through_release_rad':max((abs(s[e,t]['q'][j]-s[e,370]['q'][j]) for t in range(371,first+1) for j in (9,10)),default=0) if first else None})
o={'parity':parity,'release':rows,'note':'Observer q/v state labels lag same-run replay one frame in separately checked lanes; release flap comparison is inferred for these candidate lanes. Aero is modeled, not measured.'};(A/'parity-aero.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
