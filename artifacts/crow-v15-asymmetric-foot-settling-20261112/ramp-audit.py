from pathlib import Path
import collections,gzip,json,sys
A=Path('artifacts/crow-v15-asymmetric-foot-settling-20261112');seed=int(sys.argv[1]);D=A/f'candidate-{seed}-rebound-phase';bits={}
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if z and z[0]=='CROW_SENSOR_BITS':
   t,e=map(int,z[1:3]);assert (e,t) not in bits;bits[e,t]=int(z[3])
assert len(bits)==800
rows=[]
for e in range(2,8):
 for side in (0,1):
  cap=4 if e>=6 else 7;viol=[];counts=collections.Counter();verified=0
  for t in range(202,301):
   prev=bits[e,t-1];cur=bits[e,t]
   if not (cur&(1<<20)):continue
   before=min(((prev>>(25+3*side))&7) if prev&(1<<20) else 0,cap);rear=bool(cur&(1<<(21+2*side)));front=bool(cur&(1<<(22+2*side)));expected=min(before+2,cap) if rear and not front else max(before-1,0);actual=(cur>>(25+3*side))&7;verified+=1;counts[actual]+=1
   if actual!=expected:viol.append({'step':t,'previous':before,'rear':rear,'front':front,'expected':expected,'actual':actual})
  rows.append({'lane':e,'side':['left','right'][side],'cap':cap,'verified':verified,'level_counts':dict(counts),'violations':viol})
(A/f'ramp-audit-{seed}.json').write_text(json.dumps({'seed':seed,'rows':rows},indent=2)+'\n')
assert all(x['verified']>0 and not x['violations'] for x in rows)
for r in rows:
 if r['lane']>=4:print(seed,r['lane'],r['side'],'verified',r['verified'],'levels',r['level_counts'])
