from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-pre-release-attribution-20261113');B=Path('artifacts/crow-v15-leg-wing-attribution-20261113');P=Path('artifacts/crow-v15-launch-aero-observer-20261113')
p=json.load(gzip.open(P/'observed-window.json.gz','rt'));h=json.load(gzip.open(B/'hold-window.json.gz','rt'))
def maps(j):return {'s':{(x['lane'],x['step']):x for x in j['state']},'c':{tuple(x[:3]):x[3:] for x in j['contact']},'a':{tuple(x[:3]):x[3:] for x in j['aero']}}
p,h=maps(p),maps(h);out={}
for e in (6,7):
 d={}
 for name,x in [('push',p),('hold',h)]:
  c=x['c'];s=x['s'];seg=[];prev=None
  for t in range(365,401):
   loaded=any(sum(c[e,t,k][2:4])>1e-7 for k in range(16))
   if loaded!=prev:seg.append([t,loaded]);prev=loaded
  d[name]={'contact_transitions_through_400':seg,'foot_normal_impulse_371_374_Ns':sum(sum(c[e,t,k][:2]) for t in range(371,375) for k in range(16)),'flap_displacement_371_374_rad':max(abs(s[e,t]['q'][q]-s[e,370]['q'][q]) for t in range(371,375) for q in (9,10))}
 d['push']['modeled_aero_vertical_impulse_371_374_Ns']=sum(p['a'][e,t,k][3] for t in range(371,375) for k in range(16))*.02/16
 d['push_minus_hold_foot_normal_impulse_371_374_Ns']=d['push']['foot_normal_impulse_371_374_Ns']-d['hold']['foot_normal_impulse_371_374_Ns'];out[str(e)]=d
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
