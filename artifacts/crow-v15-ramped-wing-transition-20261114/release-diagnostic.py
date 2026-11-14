from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-ramped-wing-transition-20261114');out={}
for arm in ('push','hold'):
 j=json.load(gzip.open(A/(arm+'-window.json.gz'),'rt'));s={(x['lane'],x['step']):x for x in j['state']};c={(x[0],x[1],x[2]):x[3:] for x in j['contact']};a={(x[0],x[1],x[2]):x[3:] for x in j['aero']};out[arm]={}
 for e in (6,7):
  def loaded(t):return any(sum(c[e,t,k][2:4])>1e-7 for k in range(16))
  release=next((t for t in range(371,399) if all(not loaded(u) for u in range(t,t+3))),None)
  prior=range(371,release) if release is not None else range(371,401)
  absent=sum((e,t,k) not in a for t in prior for k in range(16))
  vertical=sum(a[e,t,k][3] for t in prior for k in range(16))*.02/16 if not absent else None
  displacement=max((abs(s[e,t]['q'][q]-s[e,370]['q'][q]) for t in prior for q in (9,10)),default=0.)
  out[arm][str(e)]={'first_three_unloaded_intervals':release,'aero_pre_release_missing':absent,'aero_pre_release_vertical_impulse_Ns':vertical,'flap_pre_release_max_displacement_rad':displacement,'pre_release_interval_bounds':[371,release-1 if release else 400]}
(A/'release-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
