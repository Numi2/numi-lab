from pathlib import Path
import gzip,json,collections
P=Path('artifacts/crow-v15-substep-gait-observer-20261111/autonomous-20261126/run.stderr.gz');A=Path('artifacts/crow-v15-gait-boundary-contact-stratification-20261112');q={};c={}
with gzip.open(P,'rt') as f:
 for l in f:
  if l.startswith('CROW_Q_SUBSTEP '):
   z=l.split();t,k,e=map(int,z[1:4]);key=e,t,k;assert key not in q;q[key]=list(map(float,z[4:]))
  elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=l.split();t,k,e=map(int,z[1:4])
   if 201<=t<=300:
    key=e,t,k;assert key not in c and int(z[4])<=int(z[5]);c[key]=list(map(float,z[6:]))
assert len(q)==len(c)==12800
bins=collections.defaultdict(list)
for e in range(8):
 for t in range(202,301):
  for k in range(16):
   def key(n):return (e,t+n//16,n%16)
   kk=[key(k-j) for j in (0,1,2)]
   for side in (0,1):
    load=[c[x][2+side]>1e-7 for x in kk]
    state='all_loaded' if all(load) else 'all_unloaded' if not any(load) else 'mixed'
    for joint,idx in [('knee',15 if side==0 else 18),('ankle',16 if side==0 else 19)]:
     acc=(q[kk[0]][idx]-2*q[kk[1]][idx]+q[kk[2]][idx])/(.00125**2)
     bins[joint,'boundary' if k==0 else 'inside',state].append(abs(acc))
rows=[]
for (joint,phase,state),vals in sorted(bins.items()):
 vals.sort();rows.append({'joint':joint,'phase':phase,'contact':state,'count':len(vals),'mean_abs_rad_s2':sum(vals)/len(vals),'median_abs_rad_s2':vals[len(vals)//2],'p95_abs_rad_s2':vals[int(.95*(len(vals)-1))]})
out={'rows':rows,'scope':'Development seed 20261126; samples pooled across 8 lanes and both sides, not independent animals. Threshold is observed foot normal impulse >1e-7 Ns. Contact category requires all three substeps in the finite difference window to be loaded or unloaded, otherwise mixed. Consecutive samples are dependent. No inference about force, biological stiffness, or causality.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r)
