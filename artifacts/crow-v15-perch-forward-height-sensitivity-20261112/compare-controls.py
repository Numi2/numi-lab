from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-perch-forward-height-sensitivity-20261112'); old=Path('artifacts/crow-v15-perch-regrip-placement-screen-20261112/high'); new=A/'high'
def records(p,kind):
 out={}
 with gzip.open(p/('trace.log.gz' if kind=='state' else 'substeps.log.gz'),'rt') as f:
  for line in f:
   z=line.split()
   if kind=='state' and z and z[0]=='step':
    e,t=map(int,z[1:3]);
    if e in ((4,5) if p==old else (0,1)):out[(e,t)]=z[z.index('q')+1:]
   if kind=='sub' and z and z[0]=='CROW_TWO_TOE_SUBSTEP':
    t,k,e=map(int,z[1:4]);
    if e in ((4,5) if p==old else (0,1)):out[(e,t,k)]=z[4:]
 return out
s0,s1=records(old,'state'),records(new,'state');c0,c1=records(old,'sub'),records(new,'sub');diff=[]
for e,ref in ((0,4),(1,5)):
 for t in range(550):
  if s0[ref,t]!=s1[e,t]:diff.append([e,t,'state'])
  for k in range(32):
   if c0[ref,t+1,k]!=c1[e,t+1,k]:diff.append([e,t,k,'contact'])
x={'matched_controls':{'0':4,'1':5},'states_compared':1100,'contact_substeps_compared':35200,'difference_count':len(diff),'differences':diff[:20]};(A/'control-parity.json').write_text(json.dumps(x,indent=2)+'\n');print(x);assert not diff
