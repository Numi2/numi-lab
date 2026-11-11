from pathlib import Path
from collections import defaultdict
import gzip,json,statistics
A=Path('artifacts/crow-v15-perch-settle-feedback-sweep-20261111'); v=defaultdict(lambda:[0.]*37);finished={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if a and a[0]=='event' and int(a[11]) and int(a[1]) not in finished:finished[int(a[1])]=int(a[2])+1
with gzip.open(A/'substeps.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if a and a[0]=='CROW_TWO_TOE_SUBSTEP':
   t,k,e=map(int,a[1:4])
   for i,x in enumerate(map(float,a[6:])):v[e,t][i]+=x
nom=.45006*9.81*.02
rows=[]
for e in range(8):
 vals=[v[e,t] for t in range(451,551)];frac=[(x[0]+x[1])/nom for x in vals]
 rows.append({'lane':e,'finished_step':finished.get(e),'last_100_mean_weight_fraction':statistics.mean(frac),'last_100_min_weight_fraction':min(frac),'last_100_max_weight_fraction':max(frac),'last_100_nonfoot_contact_intervals':sum(abs(x[4])>1e-7 for x in vals),'last_100_bilateral_opposed_intervals':sum(sum(x[21+i] for i in range(6))>=.0002 and sum(x[21+i] for i in (6,7))<=-.0002 and sum(x[29+i] for i in range(6))>=.0002 and sum(x[29+i] for i in (6,7))<=-.0002 for x in vals)})
out={'nominal_mass_kg':.45006,'rows':rows,'limitations':'Impulse-derived force fraction, no isolated aerodynamic force or momentum audit. Opposing normals alone do not establish biological grip. Applies to final two seconds of constructed bench only.'};(A/'support-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
