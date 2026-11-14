from pathlib import Path
from collections import defaultdict
import json
A=Path('artifacts/crow-v15-return-com-kinematics-20261114');d=defaultdict(dict)
lines=(A/'output.tsv').read_text().splitlines();hashes=lines[:2]
for line in lines[2:]:
 e,t,j,dx,*g=line.split();d[int(e),int(t),int(j)][float(dx)]=list(map(float,g))
ref=json.loads(Path('artifacts/crow-v15-stance-front-com-balance-20261114/geometry-analysis.json').read_text());rows=[];errors=[]
for e,key in [(0,'control'),(1,'placement')]:
 samples={x['zero_based_tick']:x for x in ref[key]['samples']}
 for t in (220,230,240,250,260,270,280,290):
  base=d[e,t,14][0.];deriv={str(j):[(a-b)/.04 for a,b in zip(d[e,t,j][.02],d[e,t,j][-.02])] for j in (14,15,16)}
  err=None
  if t in samples:
   other=[samples[t]['groups'][k]['min_box_gap_m'] for k in ('foot','front','rear')];err=max(abs(a-b) for a,b in zip(base,other));errors.append(err)
  rows.append({'lane':e,'tick':t,'base_gaps_m':base,'gap_derivative_m_per_rad_hip_knee_ankle':deriv,'independent_gap_comparison_max_error_m':err})
o={'hashes':hashes,'geometry_crosscheck_max_error_m':max(errors),'geometry_crosscheck_pass':max(errors)<1e-6,'rows':rows,'scope':'fixed-state geometry finite differences, not physical trajectories'};(A/'analysis.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
