from pathlib import Path
import gzip,json,math
D=Path('artifacts/crow-v15-release-imbalance-diagnostic-20261111');R=Path('artifacts/crow-v15-toe-regrip-velocity-damping-20261111');H=Path('artifacts/crow-v15-two-segment-grip-faster-close-perturbations-20261111')
def read(p):
 data={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for l in f:
   if l.startswith('step '):
    a=l.split();e=int(a[1]);t=int(a[2]);q=a.index('q');v=a.index('v');data[e,t]={'q':list(map(float,a[q+1:v])),'v':list(map(float,a[v+1:]))}
 return data
release=read(R);hold=read(H);r=[]
def attitude(x):
 q=x['q'];v=x['v'];qx,qy,qz,qw=q[3:7]
 return {'x_m':q[0],'y_m':q[1],'z_m':q[2],'roll_rad':math.atan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy)),'pitch_rad':math.asin(max(-1,min(1,2*(qw*qy-qz*qx)))),'linear_velocity_m_s':v[:3],'angular_velocity_rad_s':v[3:6]}
for e,h,trigger,ret,fall in [(0,0,116,152,275),(1,2,121,156,280)]:
 first=next((t for t in range(350) if release[e,t]!=hold[h,t]),None)
 sample=[]
 for t in (trigger-1,trigger,ret,ret+50,fall-1):
  sample.append({'step':t,'release':attitude(release[e,t]),'hold':attitude(hold[h,t])})
 r.append({'release_lane':e,'hold_lane':h,'first_qv_difference_step':first,'release_trigger_step':trigger,'return_step':ret,'first_root_below_0_1m':fall,'samples':sample})
(D/'analysis.json').write_text(json.dumps({'rows':r,'limitations':'Constructed stationary perch experiments, not flight arrivals; divergence is associated with release and changed release-phase controls, so the held comparison does not isolate toe opening alone. Root pose is not a measured center of pressure or stability margin.'},indent=2)+'\n')
for x in r:
 print(x['release_lane'],'firstdiff',x['first_qv_difference_step'],'trigger',x['release_trigger_step'])
 for s in x['samples']:print(s['step'],'roll/pitch release',round(s['release']['roll_rad'],3),round(s['release']['pitch_rad'],3),'hold',round(s['hold']['roll_rad'],3),round(s['hold']['pitch_rad'],3),'y release/hold',round(s['release']['y_m'],3),round(s['hold']['y_m'],3))
