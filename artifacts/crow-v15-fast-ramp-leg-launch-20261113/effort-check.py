from pathlib import Path
import gzip,json,collections
A=Path('artifacts/crow-v15-fast-ramp-leg-launch-20261113');out=[]
for arm in ('push','hold'):
 d=collections.defaultdict(lambda:[0.]*6);counts=collections.Counter();sat=collections.defaultdict(lambda:[0]*6)
 with gzip.open(A/(arm+'-wing-20261126-rebound-fastramped/run.stderr.gz'),'rt') as f:
  for l in f:
   z=l.split()
   if not z or z[0]!='CROW_LEG_EFFORT':continue
   t,k,e=map(int,z[1:4]);
   if e<2 or not 370<=t<=400:continue
   assert tuple(map(int,z[4:7]))==(1,0,0);v=list(map(float,z[7:]));assert len(v)==30
   for j in range(6):
    effort=abs(v[5*j]);d[e][j]=max(d[e][j],effort);sat[e][j]+=int(effort>=[.75,.75,.60][j%3]-1e-5)
   counts[e]+=1
 for e in range(2,8):
  assert counts[e]==31*16,(arm,e,counts[e]);bounds=[.75,.75,.60]*2
  out.append({'arm':arm,'lane':e,'max_abs_effort':d[e],'saturated_substeps':sat[e],'bound_violations':[j for j in range(6) if d[e][j]>bounds[j]+1e-5]})
assert all(not r['bound_violations'] for r in out)
(A/'effort-check.json').write_text(json.dumps({'rows':out,'scope':'Applied generalized leg effort observations in ticks 370–400; saturation and limits are model values, not measured crow strength.'},indent=2)+'\n')
for r in out:print(r)
