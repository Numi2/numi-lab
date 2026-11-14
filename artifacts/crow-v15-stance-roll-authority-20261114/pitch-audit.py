from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-stance-roll-authority-20261114');pitch={};balance={};events={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0] in ('pitchcontrol','balance','event'):
   k=(int(x[1]),int(x[2])); {'pitchcontrol':pitch,'balance':balance,'event':events}[x[0]][k]=list(map(float,x[3:]))
bad=[];active=0
for e in (0,1):
 for t in range(1,450):
  p=pitch[e,t];b=balance[e,t];last=pitch[e,t-1];event=events[e,t];limit=.14
  gate=bool(p[0]) and bool(p[1]) and (bool(event[0]) or bool(event[8]));requested=max(-limit,min(limit,-3.0*(b[2]-.004)-.1*b[4]-(.80*(p[2]-p[4])+.04*p[3]))) if gate else 0.
  applied=last[-1]+max(-.002,min(.002,requested-last[-1]));active+=bool(gate)
  if abs(requested-p[-2])>2e-7 or abs(applied-p[-1])>2e-7:bad.append([e,t,requested,p[-2],applied,p[-1]])
out={'active_samples':active,'mismatches':bad,'pass':active>0 and not bad,'scope':'Reconstructs target from logged prior balance and contact gate and post-step persistent release flag; flag is latched before target selection in source. Contact/state epoch not independently synchronized.'};(A/'pitch-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
