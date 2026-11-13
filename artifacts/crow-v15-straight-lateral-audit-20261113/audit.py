from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-straight-lateral-audit-20261113');B=Path('artifacts/crow-v15-uniform-flight-24s-screen-20261113');rows=[]
for seed in ('20261126','20261127'):
 p=B/('push-wing-012-'+seed+'-rebound-uniform');d={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e=x['lane'];t=x['step']
    if 2<=e<=7 and 500<=t<=1200:
     if (e,t) in d:raise RuntimeError('duplicate accepted state')
     d[e,t]=x
 for e in range(2,8):
  missing=[t for t in range(500,1201) if (e,t) not in d]
  if missing:rows.append({'seed':seed,'lane':e,'missing':missing});continue
  def yaw(x):
   xx,yy,zz,w=x['q'][3:7];return math.atan2(2*(w*zz+xx*yy),1-2*(yy*yy+zz*zz))
  z=[d[e,t] for t in range(500,1201)];end=d[e,1200];start=d[e,500]
  rows.append({'seed':seed,'lane':e,'missing':[],'displacement_500_1200_m':[end['q'][i]-start['q'][i] for i in range(3)],'max_abs_lateral_displacement_500_1200_m':max(abs(x['q'][1]-start['q'][1]) for x in z),'vy_500_1200_range_m_s':[min(x['v'][1] for x in z),max(x['v'][1] for x in z)],'yaw_500_1200_range_rad':[min(yaw(x) for x in z),max(yaw(x) for x in z)]})
out={'scope':'Posthoc audit of retained used-seed obstacle-free 24 s runs. The earlier preregistered screen did not impose a lateral tracking gate; this audit does not alter its historical result or qualify straight flight.','rows':rows};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r['seed'],r['lane'],'displacement',r.get('displacement_500_1200_m'),'vy',r.get('vy_500_1200_range_m_s'),'missing',len(r['missing']))
