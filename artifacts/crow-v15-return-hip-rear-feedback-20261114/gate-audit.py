from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-return-hip-rear-feedback-20261114');hip={};ctl={};evt={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if not a:continue
  if a[0]=='returnHip':hip[int(a[1]),int(a[2])]=float(a[3])
  elif a[0]=='control':ctl[int(a[1]),int(a[2])]=list(map(float,a[3:]))
  elif a[0]=='event':evt[int(a[1]),int(a[2])]=list(map(float,a[3:]))
rows=[]
for e in (0,1):
 previous=0.;bad=[];active=0
 for t in range(450):
  finished=bool(evt[e,t][8]);rear=ctl[e,t][2];goal=.01 if e==1 and finished and rear>.000025 else 0.;expected=previous+max(-.001,min(.001,goal-previous));actual=hip[e,t]
  if abs(expected-actual)>2e-8:bad.append({'tick':t,'expected':expected,'actual':actual})
  active+=goal>0;previous=actual
 rows.append({'lane':e,'ticks':450,'contact_gate_active':active,'mismatches':bad,'pass':not bad})
out={'rows':rows,'scope':'verifies control from previously recorded rear-contact observation, not same-time physical contact'};(A/'gate-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
