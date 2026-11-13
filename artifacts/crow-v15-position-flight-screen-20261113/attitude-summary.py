import gzip,json,math
p='artifacts/crow-v15-position-flight-screen-20261113/position-flight-20261126-rebound-position/run.stderr.gz'
with gzip.open(p,'rt') as f:
 for line in f:
  if not line.startswith('CROW_SUPPORT_SAMPLE '): continue
  x=json.loads(line[20:]);t=x['step'];e=x['lane']
  if e in (2,4,6) and t in (500,900,1200,1500,1800):
   xx,yy,zz,w=x['q'][3:7];n=math.sqrt(xx*xx+yy*yy+zz*zz+w*w);xx/=n;yy/=n;zz/=n;w/=n
   yaw=math.atan2(2*(w*zz+xx*yy),1-2*(yy*yy+zz*zz));roll=math.atan2(2*(w*xx+yy*zz),1-2*(xx*xx+yy*yy));print(e,t,'xyz',*[round(q,3) for q in x['q'][:3]],'roll/yaw',round(roll,3),round(yaw,3))
