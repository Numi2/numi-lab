from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-toe-release-center-screen-20261111');B=Path('artifacts/crow-v15-toe-release-regrip-20261111')
def read(path):
 d={}
 with gzip.open(path,'rt') as f:
  for line in f:
   a=line.split()
   if len(a)>3 and a[0] in ('step','event') and a[1] in ('0','1'):d[a[0],int(a[1]),int(a[2])]=line
 return d
x=read(A/'trace.log.gz');y=read(B/'trace.log.gz');rows=[]
for e in (0,1):
 keys=[(typ,e,t) for typ in ('step','event') for t in range(268)];missing=[k for k in keys if k not in x or k not in y];different=[k for k in keys if k in x and k in y and x[k]!=y[k]];rows.append({'lane':e,'missing':len(missing),'different':len(different),'pass':not missing and not different})
(A/'control-audit.json').write_text(json.dumps({'lanes':rows},indent=2)+'\n')
