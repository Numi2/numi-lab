from pathlib import Path
from collections import defaultdict
import json
A=Path('artifacts/crow-v15-return-digit-kinematics-20261114');lines=(A/'output.tsv').read_text().splitlines();d=defaultdict(dict)
for line in lines[2:]:
 e,t,j,dx,*g=line.split();d[int(e),int(t),int(j)][float(dx)]=list(map(float,g))
ref=json.loads(Path('artifacts/crow-v15-return-com-kinematics-20261114/analysis.json').read_text());base={(x['lane'],x['tick']):x['base_gaps_m'] for x in ref['rows']};rows=[];errs=[]
for e in (0,1):
 for t in (220,230,240,250,260,270,280,290):
  vals=d[e,t,101][0.];err=max(abs(a-b) for a,b in zip(vals,base[e,t]));errs.append(err);ders={str(j):[(a-b)/.04 for a,b in zip(d[e,t,j][.02],d[e,t,j][-.02])] for j in (101,102,103,104,105,106)};rows.append({'lane':e,'tick':t,'base_gaps_m':vals,'derivatives_m_per_rad':ders,'base_error_m':err})
out={'model_hashes':lines[:2],'codes':{'101':'front proximal all','102':'front distal all','103':'front both all','104':'rear proximal','105':'rear distal','106':'rear both'},'crosscheck_max_error_m':max(errs),'crosscheck_pass':max(errs)<1e-6,'rows':rows,'scope':'fixed-state finite differences; no dynamics, forces, or contacts'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
