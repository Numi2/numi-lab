from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-first-unload-wing-20261114');out={}
for arm in ('push','hold'):
 j=json.load(gzip.open(A/(arm+'-window.json.gz'),'rt'));s={(x['lane'],x['step']):x for x in j['state']};a={(x[0],x[1],x[2]):x[3:] for x in j['aero']};c={(x[0],x[1],x[2]):x[3:] for x in j['contact']};out[arm]={}
 for e in (6,7):
  first=next((t for t in range(371,401) if all(sum(c[e,t,k][2:4])<=1e-7 for k in range(16))),None);aero=sum(a[e,t,k][3] for t in range(371,first) for k in range(16))*.02/16 if first else None;flap=max((abs(s[e,t]['q'][q]-s[e,370]['q'][q]) for t in range(371,first+1) for q in (9,10)),default=0) if first else None;out[arm][str(e)]={'first_unloaded':first,'preceding_model_vertical_aero_impulse_Ns':aero,'preceding_max_flap_displacement_rad':flap}
(A/'initial-release-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
