from pathlib import Path
import gzip,json,statistics,math
paths=[Path('artifacts/crow-v15-position-flight-screen-20261113/position-flight-20261126-rebound-position/run.stderr.gz'),Path('artifacts/crow-v15-pronation-yaw-screen-20261113/pronation-yaw-20261126-rebound-pronation/run.stderr.gz')];data=[]
for p in paths:
 state={};targets={}
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line[20:]);e=x['lane'];t=x['step']
    if e in (6,7) and 650<=t<=1150:state[e,t]=x
   elif line.startswith('CROW_DRIVE_TARGET '):
    z=line.split();t=int(z[1]);e=int(z[2]);
    if e in (6,7) and 650<=t<=1150:targets[e,t]=list(map(float,z[3:]))
 data.append((state,targets))
out={}
for e in (6,7):
 samples=[]
 for t in range(650,1151):
  old=data[0][0][e,t];new=data[1][0][e,t];tar=data[1][1][e,t];samples.append((new['q'][11]-old['q'][11],new['q'][12]-old['q'][12],new['q'][11]-tar[10],new['q'][12]-tar[11],new['q'][11],new['q'][12]))
 out[e]={'samples':len(samples),'mean_actual_delta_left_right_rad':[statistics.mean(z[j] for z in samples) for j in (0,1)],'mean_abs_actual_minus_target_left_right_rad':[statistics.mean(abs(z[j]) for z in samples) for j in (2,3)],'max_abs_actual_minus_target_left_right_rad':[max(abs(z[j]) for z in samples) for j in (2,3)],'actual_position_ranges_rad':[[min(z[j] for z in samples),max(z[j] for z in samples)] for j in (4,5)]}
A=Path('artifacts/crow-v15-pronation-yaw-screen-20261113');(A/'pronation-tracking-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
