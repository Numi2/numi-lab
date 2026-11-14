from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-centered-toe-regrip-20261114');B=Path('artifacts/crow-v15-centered-stance-zero-20261114')
def read(p):
 s={};u={};finished={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for line in f:
   x=line.split()
   if not x or x[0] not in ('step','control','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','event','balance'):continue
   e,t=int(x[1]),int(x[2]);k=e,t
   if x[0]=='step':
    q=x.index('q');v=x.index('v');s[k]=list(map(float,x[q+1:v]+x[v+1:]))
   elif x[0]=='event':finished[k]=bool(int(x[11]))
   else:u.setdefault(k,[]).append((x[0],x[3:]))
 return s,u,finished
s,u,f=read(A);b,bu,_=read(B); rows=[]
for e in (0,1):
 first=next((t for t in range(450) if f.get((e,t))),None);end=first if first is not None else 450
 state_bad=[t for t in range(end) if s.get((e,t))!=b.get((1,t))]
 target_bad=[t for t in range(end) if u.get((e,t))!=bu.get((1,t))]
 rows.append({'lane':e,'first_finished_zero_based':first,'state_mismatches_before_return':len(state_bad),'first_state_mismatch':state_bad[:5],'log_mismatches_before_return':len(target_bad),'first_log_mismatch':target_bad[:5]})
o={'scope':'compares physically evolved return trial to zero-target release control before each return; logging epochs not independently verified','rows':rows};(A/'return-prefix.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
