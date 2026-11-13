from pathlib import Path
import gzip,numpy as np
r=Path(open('/tmp/crow-continuation-path').read().strip())/'repo/artifacts'
with gzip.open(r/'crow-v15-aero-sideslip-observer-20261113/samples.tsv.gz','rt') as f:s=np.loadtxt(f)
with gzip.open(r/'crow-v15-pronation-fixed-state-20261113/moments.tsv.gz','rt') as f:a=np.loadtxt(f)
p=a[4032:8064,4:];m=a[8064:12096,4:];d=(p-m)/.04
for lane in (6,7):
 mask=s[:,0]==lane
 for key,sel in [('flaprate positive',s[:,44+8]>0),('flaprate negative',s[:,44+8]<=0)]:
  x=d[mask&sel];print(lane,key,'n',len(x),'mean force',x[:,:3].mean(0),'moment',x[:,3:].mean(0),'yaw sign positive',(x[:,5]>0).mean())
