from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-gust-wrench-observer-20261114');results=[];states={}
for arm in ('control','gust'):
 p=A/('push-20261126-rebound-'+arm)
 with gzip.open(p/'replay.json.gz','rt') as f:replay=json.load(f)['payload']
 lane=replay['environment'];s={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);
    if x['lane'] in (5,6,7) and 1<=x['step']<=2400:s[x['lane'],x['step']]=x
 states[arm]=s
 match={}
 for off in (0,1):
  count=0;bad=0;maximum=0
  for frame in replay['frames']:
   x=s.get((lane,frame['step']+off))
   if x is None:continue
   err=max(abs(a-b) for a,b in zip(frame['q']+frame['v'],x['q']+x['v']));count+=1;bad+=err>1e-6;maximum=max(maximum,err)
  match[str(off)]={'compared':count,'mismatches_above_1e-6':bad,'max_absolute_error':maximum}
 results.append({'arm':arm,'replay_environment':lane,'frame_alignment':match})
def pitch(q):
 x,y,z,w=q[3:7];return math.asin(max(-1,min(1,2*(w*y-z*x))))
def wrap(x):return math.atan2(math.sin(x),math.cos(x))
rows=[]
for lane in (6,7):
 wave=[]
 for frame in range(901,1051):
  x=states['gust'].get((lane,frame+1));y=states['control'].get((lane,frame+1))
  if x and y:wave.append((frame,wrap(pitch(x['q'])-pitch(y['q'])),x['v'][4]-y['v'][4]))
 x=states['gust'].get((lane,1201));y=states['control'].get((lane,1201));dp=wrap(pitch(x['q'])-pitch(y['q'])) if x and y else None;dv=x['v'][4]-y['v'][4] if x and y else None
 pp=max((abs(v[1]) for v in wave),default=None);pv=max((abs(v[2]) for v in wave),default=None)
 rows.append({'lane':lane,'accepted_frame_window':[901,1050],'samples':len(wave),'peak_abs_pitch_delta_rad':pp,'peak_abs_world_y_rate_delta_rad_s':pv,'response_threshold_met':len(wave)==150 and (pp>=.04 or pv>=.1),'pitch_delta_at_accepted_frame_1200_rad':dp,'world_y_rate_delta_at_accepted_frame_1200_rad_s':dv,'recovery_threshold_met':dp is not None and abs(dp)<.06 and abs(dv)<.15})
out={'alignment':results,'frame_reanalysis':rows,'scope':'Replay alignment directly checked for recorded lane 5 only; offset applied to lanes 6 and 7 as a shared-host inference, not independently proven for those lanes.'};(A/'timing.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
