from pathlib import Path
import gzip,json,sys
p=Path('artifacts/crow-v15-full-flight-pronation-observer-20261114')/sys.argv[1];counts=[0]*8;sat=[0]*8;ma=[0.]*8
with gzip.open(p/'run.stderr.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x or x[0]!='CROW_PRONATION_EFFORT':continue
  t,e=int(x[1]),int(x[3]);v=list(map(float,x[4:]))
  if 390<=t<=1800 and len(v)==14:
   for j in (0,7):
    z=abs(v[j]);counts[e]+=1;sat[e]+=z>=.1499;ma[e]=max(ma[e],z)
rows=[{'lane':e,'actuator_samples':counts[e],'saturated_fraction':sat[e]/counts[e] if counts[e] else None,'max_abs_effort_Nm':ma[e]} for e in range(8)];o={'rows':rows,'scope':'full 390-1800 substep effort saturation, descriptive; no preregistered full-window fraction gate'};(p/'effort-audit.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
