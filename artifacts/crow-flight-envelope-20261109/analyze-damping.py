from pathlib import Path
import gzip,json,math,statistics
A=Path('artifacts/crow-flight-envelope-20261109')
names=['yaw-negative','yaw-zero','damping-0.15-yaw-negative','damping-0.15-yaw-zero','damping-0.05-yaw-negative','damping-0.05-yaw-zero']
out={}
for name in names:
 raw=json.loads((A/(name+('-analysis.json' if name.startswith('yaw-') else '-results.json'))).read_text())
 first=raw['first_episode'];samples=[json.loads(t.split(' ',1)[1]) for t in gzip.open(A/(name+'.stderr.gz'),'rt') if t.startswith('CROW_FLIGHT_SAMPLE ')]
 lanes=[]
 for lane in range(32):
  xs=sorted((x for x in samples if x['lane']==lane and 1000<=x['step']<=1500 and x['step']<=first['steps_by_environment'][lane]),key=lambda x:x['step'])
  if len(xs)!=6:
   lanes.append({'lane':lane,'samples':len(xs),'physical_failure':first['physical_failure_by_environment'][lane]});continue
  yaw=lambda x:math.atan2(2*(x['q'][6]*x['q'][5]+x['q'][3]*x['q'][4]),1-2*(x['q'][4]**2+x['q'][5]**2))
  angles=[yaw(x) for x in xs];rate=sum(math.atan2(math.sin(b-a),math.cos(b-a)) for a,b in zip(angles,angles[1:]))/10
  v=xs[-1]['v'];f=math.cos(angles[-1])*v[0]+math.sin(angles[-1])*v[1]
  lanes.append({'lane':lane,'samples':len(xs),'physical_failure':first['physical_failure_by_environment'][lane],'mean_yaw_rate_rad_s':rate,'forward_speed_at_30s_m_s':f,'root_height_at_30s_m':xs[-1]['q'][2]})
 complete=all(r['samples']==6 for r in lanes)
 fields=['mean_yaw_rate_rad_s','forward_speed_at_30s_m_s','root_height_at_30s_m']
 out[name]={'complete':complete,'physical_failure_count':sum(r['physical_failure'] for r in lanes),'failed_environment_steps':raw['failed_environment_steps'],'median':{k:statistics.median(r[k] for r in lanes) for k in fields} if complete else None,'lanes':lanes}
(A/'damping-summary.json').write_text(json.dumps(out,indent=2)+'\n')
for name,v in out.items():print(name,v['physical_failure_count'],v['median'])
