from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-double-disturbance-20261114');B=Path('artifacts/crow-v15-sustained-disturbance-20261114');data={}
for arm in ('control','gust'):
 s={}
 with gzip.open((B if arm=='control' else A)/(arm+'-20261127-rebound-observer-'+arm)/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);
    if x['lane'] in (6,7) and 902<=x['step']<=1201:s[x['lane'],x['step']]=x
 data[arm]=s
def pitch(q):
 x,y,z,w=q[3:7];return math.asin(max(-1,min(1,2*(w*y-z*x))))
def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi
rows=[]
for e in (6,7):
 wave=[]
 for frame in range(901,1051):
  x=data['gust'][e,frame+1];y=data['control'][e,frame+1];wave.append((frame,wrap(pitch(x['q'])-pitch(y['q'])),x['v'][4]-y['v'][4]))
 x=data['gust'][e,1201];y=data['control'][e,1201];pp=max(abs(v[1]) for v in wave);pv=max(abs(v[2]) for v in wave);dp=wrap(pitch(x['q'])-pitch(y['q']));dv=x['v'][4]-y['v'][4]
 rows.append({'lane':e,'accepted_frame_window':[901,1050],'peak_abs_pitch_delta_rad':pp,'peak_abs_world_y_rate_delta_rad_s':pv,'response_threshold_met':pp>=.04 or pv>=.1,'pitch_delta_at_accepted_frame_1200_rad':dp,'world_y_rate_delta_at_accepted_frame_1200_rad_s':dv,'recovery_threshold_met':abs(dp)<.06 and abs(dv)<.15})
out={'rows':rows,'scope':'supplementary frame-labelled reanalysis; offset inferred from shared host but must be independently checked in this run; original preregistered analysis retained'};(A/'timing-recheck.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
