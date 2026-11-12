from pathlib import Path
import gzip,json,math,statistics
A=Path('artifacts/crow-v15-speed-feedback-long-flight-20261112');p=A/'candidate-20261127-rebound-flexed/run.stderr.gz';lanes=[[] for _ in range(8)]
with gzip.open(p,'rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);
   if 601<=x['step']<=1200:lanes[x['lane']].append(x)
out=[]
for e,x in enumerate(lanes):
 assert len(x)==600
 def rpy(q):
  a,b,c,w=q[3:7];return (math.atan2(2*(w*a+b*c),1-2*(a*a+b*b)),math.asin(max(-1,min(1,2*(w*b-c*a)))),math.atan2(2*(w*c+a*b),1-2*(b*b+c*c)))
 y=[v['q'][1] for v in x];z=[v['q'][2] for v in x];vx=[v['v'][0] for v in x];vy=[v['v'][1] for v in x];angles=[rpy(v['q']) for v in x]
 out.append({'lane':e,'height_range_m':[min(z),max(z)],'mean_world_vx_m_s':statistics.mean(vx),'mean_abs_world_vy_m_s':statistics.mean(map(abs,vy)),'maximum_abs_roll_rad':max(abs(a[0]) for a in angles),'maximum_abs_pitch_rad':max(abs(a[1]) for a in angles),'net_lateral_displacement_m':y[-1]-y[0]})
(A/'flight-summary.json').write_text(json.dumps({'interval_s':[12,24],'rows':out,'scope':'Endpoint sampling at 50 Hz; world-frame velocities are not body-relative speed tracking error, and extrema are sampled.'},indent=2)+'\n')
for z in out:print(z['lane'],'z',*[round(v,3) for v in z['height_range_m']],'vxmean',round(z['mean_world_vx_m_s'],3),'abs vy',round(z['mean_abs_world_vy_m_s'],3),'roll/pitch',round(z['maximum_abs_roll_rad'],3),round(z['maximum_abs_pitch_rad'],3),'dy',round(z['net_lateral_displacement_m'],3))
