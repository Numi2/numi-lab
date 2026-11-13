from pathlib import Path
import gzip,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');A=R/'crow-v15-continuous-push-phase-audit-20261113';P=R/'crow-v15-continuous-push-transition-20261113';d={}
for n in ('push-wing','hold-wing'):
 p=P/(n+'-20261126-rebound-continuous/run.stderr.gz');records={}
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 348<=t<=357:records[e,t,k]=sum(map(float,z[6:8]))
 assert len(records)==8*10*16
 d[n]=records
rows=[]
for e in range(2,8):
 row={'lane':e}
 for label,lo,hi in (('early',348,355),('late',356,357),('whole',348,357)):
  row[label+'_delta_ns']=sum(d['push-wing'][e,t,k]-d['hold-wing'][e,t,k] for t in range(lo,hi+1) for k in range(16))
 rows.append(row)
(A/'analysis.json').write_text(json.dumps({'scope':'Posthoc, one used seed and correlated lanes; phase boundaries were selected after inspecting data. Wing targets begin moving near the boundary; impulse differences are coupled closed-loop effects, not isolated leg or muscle forces.','rows':rows},indent=2)+'\n')
for x in rows:print(x)
