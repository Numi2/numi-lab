from pathlib import Path
import gzip,json,collections,math
P=Path('artifacts/crow-v15-pretransfer-pitch-feedback-20261114');A=Path('artifacts/crow-v15-return-load-transfer-audit-20261114');V=collections.defaultdict(lambda:[0.,0.]);coverage=collections.defaultdict(set);duplicates=[]
with gzip.open(P/'substeps.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if z and z[0]=='CROW_TWO_TOE_SUBSTEP':
   t,s,e=map(int,z[1:4]);k=(e,t)
   if s in coverage[k]:duplicates.append([e,t,s])
   coverage[k].add(s);V[k][0]+=float(z[6]);V[k][1]+=float(z[7])
rows=[]
for e in (0,1):
 vals=[]
 for t in range(285,321):
  l,r=V[e,t];tot=l+r;frac=l/tot if abs(tot)>1e-6 else None;vals.append({'observer_interval':t,'left_total_upward_impulse_Ns':l,'right_total_upward_impulse_Ns':r,'left_fraction_of_total':frac,'substep_count':len(coverage[e,t])})
 jumps=[(abs(vals[i]['left_fraction_of_total']-vals[i-1]['left_fraction_of_total']),vals[i-1]['observer_interval'],vals[i]['observer_interval']) for i in range(1,len(vals)) if vals[i]['left_fraction_of_total'] is not None and vals[i-1]['left_fraction_of_total'] is not None]
 rows.append({'lane':e,'largest_adjacent_fraction_change':max(jumps) if jumps else None,'intervals':vals})
out={'rows':rows,'duplicates':duplicates,'coverage_complete':all(v['substep_count']==32 for r in rows for v in r['intervals']) and not duplicates,'scope':'post-hoc summed normal plus tangential vertical contact impulse by foot; fractions can be negative or exceed one when a foot pulls downward. Observer control-step alignment with accepted q and last-substep controller contact remains unverified; correlation is not a causal conclusion.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
