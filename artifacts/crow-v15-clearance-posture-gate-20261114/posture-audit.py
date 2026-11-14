from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-clearance-posture-gate-20261114');balance={};stage={};pose={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  a=line.split()
  if a and a[0] in ('balance','stage','placementGate'):
   k=int(a[1]),int(a[2]);{'balance':balance,'stage':stage,'placementGate':pose}[a[0]][k]=list(map(float,a[3:]))
bad=[]
for e in (0,1):
 for t in range(450):
  b=balance[e,t];p=pose[e,t];s=stage[e,t];x,y,vx,vy=b[2:6];tilt=p[1]
  expected=(abs(x)<.015 and abs(y)<.030 and abs(vx)<.05 and abs(vy)<.05) if e==0 else (abs(x)<.040 and abs(y)<.035 and abs(vx)<.08 and abs(vy)<.08 and tilt<.35)
  if expected!=bool(s[2]):bad.append([e,t,expected,bool(s[2])])
out={'mismatches':bad,'pass':not bad,'scope':'Reconstructs gates from logged model-relative COM and tilt; state/contact epoch not independently verified.'};(A/'posture-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
