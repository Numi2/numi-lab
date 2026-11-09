from pathlib import Path
from collections import defaultdict
import json
A=Path('artifacts/crow-v15-grip-search-20261109');lines=(A/'perch-perturb-results.txt').read_text().splitlines();C=defaultdict(float);phase={};results={}
for line in lines:
 a=line.split()
 if a[0]=='contact':
  c,t=int(a[1]),int(a[2]);name=a[3];nx=float(a[4]);imp=float(a[6]);side='left' if '_left_' in name else 'right';digit='rear' if '_rear_' in name else 'front';direction='positive' if nx>.5 else 'negative' if nx<-.5 else 'other';C[c,t,side,digit,direction]+=imp
 elif a[0]=='trace':phase[int(a[1]),int(a[2])]=int(a[3])
 elif a[0]=='case_result':results[int(a[1])]=line
rows=[]
for c in range(8):
 hits=[t for t in range(150) if phase.get((c,t))==3 and all(C[c,t,side,'front','negative']>.00005 and C[c,t,side,'rear','positive']>.00005 for side in ('left','right'))]
 run=max((len(list(g)) for _,g in __import__('itertools').groupby(enumerate(hits),lambda q:q[1]-q[0])),default=0)
 release=next((t for t in range(150) if phase.get((c,t))==4),None)
 loss={side:next((t for t in range(release or 150,150) if C[c,t,side,'front','negative']<=.00005 or C[c,t,side,'rear','positive']<=.00005),None) for side in ('left','right')}
 passed=run>=3 and release is not None and all(v is not None for v in loss.values()) and 'completed' in results.get(c,'')
 rows.append({'case':c,'matched_result':results.get(c,'missing'),'both_feet_opposing_hold_steps':hits,'max_consecutive_hold':run,'release_step':release,'post_release_loss':loss,'passes':passed})
out={'cases':rows,'pass_count':sum(x['passes'] for x in rows),'scope':'eight initial perturbations on isolated estimated toe body; no dynamic perturbation, landing or whole-sequence qualification'};(A/'perch-perturb-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
