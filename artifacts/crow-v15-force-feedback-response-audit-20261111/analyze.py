from pathlib import Path
from collections import defaultdict
import gzip,json,hashlib
D=Path('artifacts/crow-v15-force-feedback-response-audit-20261111');P=Path('artifacts/crow-v15-regrip-force-feedback-20261111/results.txt.gz');selected=(1,2,5,8,11,14);h=defaultdict(dict);tr={};feet={}
with gzip.open(P,'rt') as f:
 for l in f:
  v=l.split()
  if not v or v[0] not in ('toe_horizontal','trace','foot_interval'):continue
  e,t=int(v[1]),int(v[3])
  if e not in selected or not 877<=t<1500:continue
  if v[0]=='toe_horizontal':h[e,t][int(v[4])]=float(v[5])
  elif v[0]=='trace':tr[e,t]=list(map(float,v[4:]))
  else:feet[e,t]=list(map(float,v[4:6]))
def mean(a):return sum(a)/len(a)
rows=[]
for e in selected:
 for a,b in ((877,925),(925,1000),(1000,1250),(1400,1500)):
  H=[h[e,t] for t in range(a,b)];Q=[tr[e,t] for t in range(a,b)];F=[feet[e,t] for t in range(a,b)];assert all(len(q)==8 for q in H)
  rows.append({'lane':e,'window':[a,b],'mean_signed_opposing_horizontal_impulse_Ns':{'left_front':mean([sum(x[k] for k in (0,1,2)) for x in H]),'left_rear':mean([-x[3] for x in H]),'right_front':mean([sum(x[k] for k in (4,5,6)) for x in H]),'right_rear':mean([-x[7] for x in H])},'mean_rear_toe_angles_rad':[mean([x[16] for x in Q]),mean([x[20] for x in Q])],'mean_foot_upward_impulses_Ns':[mean([x[0] for x in F]),mean([x[1] for x in F])]})
out={'source_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'scope':'post-hoc descriptive means only; no proof of causation','rows':rows};(D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('retained',len(rows),'window summaries')
