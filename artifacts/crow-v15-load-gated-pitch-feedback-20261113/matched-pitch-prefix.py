from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-load-gated-pitch-feedback-20261113')
B=Path('artifacts/crow-v15-supported-pitch-feedback-20261113')
def read(p):
 out={}
 with gzip.open(p/'trace.log.gz','rt') as f:
  for s in f:
   z=s.split()
   if len(z)>3 and z[0] in ('step','pitchcontrol'):
    out[z[0],int(z[1]),int(z[2])]=z[3:]
 return out
a,b=read(A),read(B);r={}
for e in range(2):
 changed=[];state=[]
 for t in range(450):
  x=a.get(('pitchcontrol',e,t));y=b.get(('pitchcontrol',e,t))
  if x and y and (x[-2:]!=y[-2:]):changed.append(t)
  x=a.get(('step',e,t));y=b.get(('step',e,t))
  if x!=y:state.append(t)
 first=changed[0] if changed else None
 r[str(e)]={'first_command_difference_zero_based':first,'first_state_difference_zero_based':state[0] if state else None,'identical_states_before_command_change':all(t not in state for t in range(first or 0)),'complete_state_records':sum(('step',e,t) in a for t in range(450))}
(A/'matched-pitch-prefix.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
