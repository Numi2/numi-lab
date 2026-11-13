from pathlib import Path
import collections,gzip,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-ankle-push-reach-20261113';p=A/'push-nowings-20261127-rebound-ankle/run.stderr.gz';events=[];counts=collections.Counter()
with gzip.open(p,'rt') as f:
 for line in f:
  if line.startswith('CROW_TOE_V_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));assert len(v)==16
   if e>=2 and 348<=t<=380:
    for j,x in enumerate(v):
     if abs(x)>=11.99:events.append({'lane':e,'step':t,'substep':k,'toe_dof_index':j,'velocity_rad_s':x});counts[j]+=1
out={'threshold_rad_s':11.99,'configured_toe_velocity_limit_rad_s':12.,'events':events,'counts_by_toe_dof':dict(counts),'scope':'Posthoc recorded toe coordinate speeds at or above threshold, not a direct record of whether the solver actively clipped a velocity. Indices 0–7 left, 8–15 right: four proximal/distal pairs. Correlated lanes on one used seed.'};(A/'toe-events.json').write_text(json.dumps(out,indent=2)+'\n');print('counts',counts,'range',min([x['step'] for x in events]),max([x['step'] for x in events]))
