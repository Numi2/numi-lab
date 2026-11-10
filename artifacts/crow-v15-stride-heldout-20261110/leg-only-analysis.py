from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-stride-heldout-20261110');seed=709815427;data={}
for arm in ('push','control'):
 samples={}
 with gzip.open(A/f'{seed}-{arm}.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);t=x['step']
    if t in (1,650,653,654,655):samples[x['lane'],t]=x
 data[arm]=samples
out=[]
for lane in range(8):
 p=data['push'];c=data['control'];base=p[lane,650];end=p[lane,654];cb=c[lane,650];ce=c[lane,654];initial=max(abs(x-y) for field in ('q','v') for x,y in zip(p[lane,1][field],c[lane,1][field]));prefix=max(abs(x-y) for field in ('q','v') for x,y in zip(base[field],cb[field]));rise=end['q'][2]-base['q'][2];crise=ce['q'][2]-cb['q'][2];wing=max(abs(end['q'][k]-base['q'][k]) for k in (9,10));out.append({'lane':lane,'initial_max_qv_difference':initial,'prepush_max_qv_difference':prefix,'push_rise_m':rise,'control_rise_m':crise,'extra_rise_m':rise-crise,'pre_latch_wing_angle_change_rad':wing,'push_final_vz':end['v'][2]})
r={'seed':seed,'sampling':'accepted states at control steps 650 and 654; latch is reported at task step 654; substep details not reconstructed','lanes':out};(A/'leg-only-analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(out,indent=2))
