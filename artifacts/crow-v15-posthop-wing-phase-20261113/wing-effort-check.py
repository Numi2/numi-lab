from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-posthop-wing-phase-20261113');rows=[]
for arm in ('push','hold'):
 m=collections.defaultdict(lambda:[0.,0.]);counts=collections.Counter()
 with gzip.open(A/(arm+'-wing-20261126-rebound-posthop/run.stderr.gz'),'rt') as f:
  for l in f:
   z=l.split()
   if not z or z[0]!='CROW_WING_EFFORT':continue
   t,k,e=map(int,z[1:4]);
   if e<2 or not 348<=t<=500:continue
   v=list(map(float,z[4:]));assert len(v)==14
   for j in (0,1):m[e][j]=max(m[e][j],abs(v[7*j]))
   counts[e]+=1
 for e in range(2,8):
  assert counts[e]==153*16;rows.append({'arm':arm,'lane':e,'max_abs_flap_effort':m[e],'within_3_bound':all(x<=3.00001 for x in m[e])})
(A/'wing-effort-check.json').write_text(json.dumps({'rows':rows,'scope':'Flap actuators only; sweep effort is not measured by this observer.'},indent=2)+'\n')
for x in rows:print(x)
