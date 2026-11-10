from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-third-substep-20261110');name=sys.argv[1];release=int(sys.argv[2]);start=int(sys.argv[3]);end=int(sys.argv[4]);status=set();contacts={}
with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
 for line in f:
  if line.startswith('CROW_SUBSTEP_STATUS '):
   _,t,sub,lane,_,_=line.split();status.add((int(t),int(sub),int(lane)))
  elif line.startswith(('CROW_SUBSTEP_TOE ','CROW_SUBSTEP_FOOT ')):
   label,t,sub,lane,a,b,imp,sep,nz,nx,ny=line.split();key=(int(t),int(sub),int(lane));a=int(a);b=int(b);body=a if a!=30 else b;side='left' if body in (10,17,18,19,20) else 'right';p=abs(float(imp));d=contacts.setdefault(key,{'left_toe':0.,'right_toe':0.,'left_up':0.,'right_up':0.})
   if label=='CROW_SUBSTEP_TOE':d[side+'_toe']+=p
   d[side+'_up']+=max((-1 if body==a else 1)*float(nz)*p,0.)
expected={(t,s,0) for t in range(start,end+1) for s in range(4)}
def longest(seq):
 m=n=0
 for x in seq:n=n+1 if x else 0;m=max(m,n)
 return m
out={'release_step':release,'trace_start':start,'trace_end':end,'expected_substeps':len(expected),'observed_substeps':len(expected&status),'missing_substeps':len(expected-status),'sides':{}}
for side,other in [('left','right'),('right','left')]:
 lo,hi=(release+10,release+60) if side=='left' else (release+70,release+120);keys=sorted(k for k in expected if lo<=k[0]<=hi);good=[]
 for key in keys:
  x=contacts.get(key,{'left_toe':0.,'right_toe':0.,'left_up':0.,'right_up':0.});good.append(key in status and x[side+'_toe']<=1e-8 and x[other+'_up']>.0000125)
 out['sides'][side]={'window_substeps':len(keys),'qualifying_substeps':sum(good),'longest_consecutive_substeps':longest(good),'first_qualifying':next((list(k) for k,g in zip(keys,good) if g),None)}
out['limitations']='Substep contact impulses; control-step index alignment is checked against accepted trajectory separately. No substep geometric shape clearance or biological validation.';(A/(name+'-substep-audit.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
