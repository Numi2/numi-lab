from pathlib import Path
import gzip,json
A=Path(__file__).resolve().parent;data={}
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if x and x[0]=='balance':data[int(x[1]),int(x[2])]=list(map(float,x[3:]))
bad=[];active=0
for e in (0,1):
 for t in range(1,450):
  z=data[e,t];prev=data[e,t-1];old=prev[-1];expected=old
  if z[0]:
   desired=max(-.25,min(.25,4*(z[3]-(.005 if e==1 else .01)+.1*z[5])/.13));expected=old+max(-.005,min(.005,desired-old));active+=1
  if abs(expected-z[-1])>3e-7:bad.append([e,t,expected,z[-1]])
out={'active_samples':active,'mismatches':bad,'pass':active>0 and not bad,'scope':'reconstructs bounded roll target from logged previous-state balance feedback, not physical effort'};(A/'roll-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
